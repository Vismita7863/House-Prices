import pandas as pd
import numpy as np
import warnings
import optuna

from sklearn.linear_model import Ridge
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p
from scipy.optimize._optimize import BracketError

warnings.filterwarnings('ignore')

print("\n=== AmesHousing: Tuned Ensemble with XGB, Ridge, CatBoost, LGB ===")

print("\n[1/5] Loading and preparing data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
ames_df = pd.read_csv('AmesHousing.csv')  
test_ids = test_df['Id']

ames_df.columns = ames_df.columns.str.replace(' ', '').str.replace('/', '')
train_df.columns = train_df.columns.str.replace(' ', '').str.replace('/', '')
test_df.columns = test_df.columns.str.replace(' ', '').str.replace('/', '')
ames_df = ames_df.rename(columns={'PID': 'Id'}).drop('Order', axis=1)

full_train_df = pd.concat([train_df, ames_df.loc[~ames_df['Id'].isin(test_df['Id'])]]).reset_index(drop=True)
full_train_df.drop_duplicates(inplace=True)

full_train_df = full_train_df.drop(full_train_df[(full_train_df['GrLivArea'] > 4000) & 
                                                 (full_train_df['SalePrice'] < 300000)].index).reset_index(drop=True)

y = np.log1p(full_train_df["SalePrice"])
full_train_df = full_train_df.drop(['Id','SalePrice'], axis=1)
test_df = test_df.drop('Id', axis=1)
all_data = pd.concat([full_train_df, test_df]).reset_index(drop=True)

for col in ('PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType','GarageFinish',
            'GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','MasVnrType'): all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
            'TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea'): all_data[col] = all_data[col].fillna(0)
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
for col in ('MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType','Utilities','Functional'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + 0.5*all_data['HalfBath'] + all_data['BsmtFullBath'] + 0.5*all_data['BsmtHalfBath']
all_data['TotalPorchSF'] = all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF']
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
all_data['Has2ndFlr'] = (all_data['2ndFlrSF'] > 0).astype(int)
all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
all_data['HasBsmt'] = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)

ordinal_map = {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5,'No':1,'Mn':2,'Av':3,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6,'RFn':2,'Fin':3}
cols_ordinal = ['ExterQual','ExterCond','BsmtQual','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC','BsmtExposure','BsmtFinType1','BsmtFinType2','GarageFinish']
for col in cols_ordinal: all_data[col] = all_data[col].map(ordinal_map).fillna(0)
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

numerical_feats = all_data.select_dtypes(include=np.number).columns
skewed_feats = all_data[numerical_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[abs(skewed_feats) > 0.5].index
for feat in skewed_feats:
    try: all_data[feat] = boxcox1p(all_data[feat], boxcox_normmax(all_data[feat]+1))
    except (ValueError, BracketError): all_data[feat] = np.log1p(all_data[feat])

final_data = pd.get_dummies(all_data).reset_index(drop=True)
final_data.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(strategy='median')
final_data = pd.DataFrame(imputer.fit_transform(final_data), columns=imputer.get_feature_names_out())

X = final_data.iloc[:len(y)]
X_test = final_data.iloc[len(y):]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
print("Data preparation complete.")

def rmsle_cv(model, X, y, cv=kfold):
    scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1))
    return scores.mean()

print("\n[2/5] Tuning all models (XGBoost first)...")

def objective_xgb(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42, "n_jobs": -1
    }
    model = xgb.XGBRegressor(**params)
    return rmsle_cv(model, X_scaled, y)

study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=50)
print(f"Best XGBoost CV RMSLE: {study_xgb.best_value:.5f}")

def objective_ridge(trial):
    alpha = trial.suggest_float('alpha', 1e-2, 100, log=True)
    model = Ridge(alpha=alpha, random_state=42)
    return rmsle_cv(model, X_scaled, y)

study_ridge = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_ridge.optimize(objective_ridge, n_trials=50)
print(f"Best Ridge CV RMSLE: {study_ridge.best_value:.5f}")

def objective_catboost(trial):
    params = {'iterations': trial.suggest_int('iterations', 1000, 5000),
              'depth': trial.suggest_int('depth', 4, 8),
              'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
              'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True)}
    model = cb.CatBoostRegressor(**params, loss_function='RMSE', verbose=0, random_seed=42)
    return rmsle_cv(model, X, y)

study_cat = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_cat.optimize(objective_catboost, n_trials=50)
print(f"Best CatBoost CV RMSLE: {study_cat.best_value:.5f}")

def objective_lgb(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 80),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42, "n_jobs": -1, "verbose": -1
    }
    model = lgb.LGBMRegressor(**params)
    return rmsle_cv(model, X, y)

study_lgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(objective_lgb, n_trials=50)
print(f"Best LightGBM CV RMSLE: {study_lgb.best_value:.5f}")

print("\n[3/5] Training final models...")
best_xgb = xgb.XGBRegressor(**study_xgb.best_params, random_state=42, n_jobs=-1)
best_ridge = Ridge(**study_ridge.best_params, random_state=42)
best_cat = cb.CatBoostRegressor(**study_cat.best_params, loss_function='RMSE', verbose=0, random_seed=42)
best_lgb = lgb.LGBMRegressor(**study_lgb.best_params, random_state=42, n_jobs=-1, verbose=-1)

best_xgb.fit(X_scaled, y)
best_ridge.fit(X_scaled, y)
best_cat.fit(X, y)
best_lgb.fit(X, y)

print("\n[4/5] Generating predictions and ensemble...")

pred_xgb = best_xgb.predict(X_test_scaled)
pred_ridge = best_ridge.predict(X_test_scaled)
pred_cat = best_cat.predict(X_test)
pred_lgb = best_lgb.predict(X_test)

pred_xgb = np.expm1(pred_xgb)
pred_ridge = np.expm1(pred_ridge)
pred_cat = np.expm1(pred_cat)
pred_lgb = np.expm1(pred_lgb)

pd.DataFrame({'Id': test_ids, 'SalePrice': pred_xgb}).to_csv('final_submission_xgb.csv', index=False)
pd.DataFrame({'Id': test_ids, 'SalePrice': pred_ridge}).to_csv('final_submission_ridge.csv', index=False)
pd.DataFrame({'Id': test_ids, 'SalePrice': pred_cat}).to_csv('final_submission_catboost.csv', index=False)
pd.DataFrame({'Id': test_ids, 'SalePrice': pred_lgb}).to_csv('final_submission_lgb.csv', index=False)

weights = np.array([1/study_xgb.best_value, 1/study_ridge.best_value, 1/study_cat.best_value, 1/study_lgb.best_value])
weights = weights / weights.sum()

pred_ensemble_log = (weights[0]*np.log1p(pred_xgb) + 
                     weights[1]*np.log1p(pred_ridge) + 
                     weights[2]*np.log1p(pred_cat) + 
                     weights[3]*np.log1p(pred_lgb))

pred_ensemble = np.expm1(pred_ensemble_log)
pd.DataFrame({'Id': test_ids, 'SalePrice': pred_ensemble}).to_csv('final_submission_ensemble.csv', index=False)

print("✅ All submissions created including ENSEMBLE.")
print("=== Script Complete ===")
