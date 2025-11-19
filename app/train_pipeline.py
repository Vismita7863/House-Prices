import os
import sys
import argparse
import warnings
import joblib
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    print("WARNING: MLflow not installed. Experiment tracking will be skipped.")
    MLFLOW_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

SEED = 42
N_FOLDS = 5
EXPERIMENT_NAME = "Ames_Housing_Production"
TRACKING_URI = "file:./mlruns"

ORDINAL_MAPPINGS = {
    "ExterQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
    "ExterCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
    "BsmtQual":  {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
    "BsmtCond":  {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
    "KitchenQual":{"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
    "HeatingQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
    "GarageQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
    "GarageCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
    "FireplaceQu":{"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
}

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(debug=False):
    """Loads train and test data from the sibling 'data' directory."""
    train_path = os.path.join('..', 'data', 'train.csv')
    test_path = os.path.join('..', 'data', 'test.csv')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.error(f"Data files not found at {train_path}. Check your folder structure.")
        sys.exit(1)
        
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    if debug:
        logger.info("Running in DEBUG mode (using first 100 rows).")
        train = train.head(100)
        test = test.head(100)
        
    train = train[train.GrLivArea < 4500].reset_index(drop=True)
    logger.info(f"Data Loaded. Train shape: {train.shape}, Test shape: {test.shape}")
    return train, test

def feature_engineering(df):
    df = df.copy()
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    if 'TotalBsmtSF' in df.columns and '1stFlrSF' in df.columns and '2ndFlrSF' in df.columns:
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    if 'YearBuilt' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    return df

def get_preprocessor(X_train):
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=20)) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose=False
    )
    return preprocessor

def get_model(model_name, params=None):
    if params is None: params = {}
    
    if model_name == 'lightgbm':
        if not LGBM_AVAILABLE: raise ImportError("LightGBM requested but not installed.")
        default_params = {'random_state': SEED, 'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 31, 'verbose': -1}
        default_params.update(params)
        return lgb.LGBMRegressor(**default_params)
    elif model_name == 'xgboost':
        if not XGB_AVAILABLE: raise ImportError("XGBoost requested but not installed.")
        default_params = {'random_state': SEED, 'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 6}
        default_params.update(params)
        return xgb.XGBRegressor(**default_params)
    elif model_name == 'rf':
        default_params = {'random_state': SEED, 'n_estimators': 200, 'max_depth': 15}
        default_params.update(params)
        return RandomForestRegressor(**default_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_optuna(X, y, model_type):
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not installed. Skipping tuning.")
        return {}
    logger.info(f"Starting Optuna tuning for {model_type}...")
    def objective(trial):
        if model_type == 'lightgbm':
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                'random_state': SEED, 'verbose': -1
            }
            model = lgb.LGBMRegressor(**param)
        else: return 0
        kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
        return scores.mean()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    return study.best_params

def generate_plots(model, X_encoded, feature_names):
    plots = {}
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]
            plt.figure(figsize=(10, 6))
            plt.title("Top 20 Feature Importances")
            plt.barh(range(len(indices)), importances[indices], align='center')
            if feature_names is not None and len(feature_names) == len(importances):
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            else:
                plt.yticks(range(len(indices)), indices)
            plt.xlabel("Relative Importance")
            plt.tight_layout()
            fi_path = "feature_importance.png"
            plt.savefig(fi_path)
            plots['feature_importance'] = fi_path
            plt.close()
    except Exception as e:
        logger.warning(f"Could not generate feature importance plot: {e}")

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            X_sample = X_encoded[:200]
            shap_values = explainer.shap_values(X_sample)
            plt.figure()
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            shap_path = "shap_summary.png"
            plt.savefig(shap_path, bbox_inches='tight')
            plots['shap_summary'] = shap_path
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate SHAP plot: {e}")
    return plots

def main():
    parser = argparse.ArgumentParser(description="Ames Housing Training Pipeline")
    parser.add_argument("--model", type=str, default="lightgbm", choices=["lightgbm", "xgboost", "rf"], help="Model type")
    parser.add_argument("--optuna", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--debug", action="store_true", help="Run on a small subset")
    parser.add_argument("--train-only", action="store_true", help="Skip submission generation")
    args = parser.parse_args()
    
    if args.model == 'lightgbm' and not LGBM_AVAILABLE:
        logger.warning("LightGBM not available, falling back to RandomForest.")
        args.model = 'rf'

    logger.info("Step 1: Loading and Preparing Data")
    train_df, test_df = load_data(debug=args.debug)
    test_ids = test_df['Id']
    
    train_fe = feature_engineering(train_df)
    test_fe = feature_engineering(test_df)
    
    y = np.log1p(train_fe['SalePrice'])
    X = train_fe.drop(['SalePrice'], axis=1, errors='ignore')
    X_test = test_fe
    
    logger.info("Step 2: Defining Preprocessing")
    preprocessor = get_preprocessor(X)
    
    best_params = {}
    if args.optuna:
        logger.info("Running Optuna on temporary transformed data...")
        X_temp = preprocessor.fit_transform(X)
        best_params = run_optuna(X_temp, y, args.model)

    logger.info(f"Step 3: Training {args.model} with MLflow tracking")
    
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        run_context = mlflow.start_run()
    else:
        from contextlib import nullcontext
        run_context = nullcontext()

    with run_context:
        if MLFLOW_AVAILABLE:
            mlflow.log_param("model_type", args.model)
            mlflow.log_param("seed", SEED)
            if best_params: mlflow.log_params(best_params)

        model_regressor = get_model(args.model, params=best_params)
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model_regressor)
        ])

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        cv_scores = cross_val_score(full_pipeline, X, y, cv=kf, scoring='neg_root_mean_squared_error')
        
        mean_rmse = -cv_scores.mean()
        std_rmse = cv_scores.std()
        logger.info(f"CV RMSE (log scale): {mean_rmse:.4f} +/- {std_rmse:.4f}")
        
        if MLFLOW_AVAILABLE:
            mlflow.log_metric("cv_rmse_mean", mean_rmse)

        logger.info("Fitting pipeline on full dataset...")
        full_pipeline.fit(X, y)
        
        model_path = "model_pipeline.pkl"
        joblib.dump(full_pipeline, model_path)
        logger.info(f"Model saved to: {os.path.abspath(model_path)}")
        
        if MLFLOW_AVAILABLE:
            input_example = X.head()
            prediction_example = full_pipeline.predict(input_example)
            signature = infer_signature(input_example, prediction_example)
            mlflow.sklearn.log_model(full_pipeline, "model", signature=signature, input_example=input_example)
                
        try:
            fitted_preprocessor = full_pipeline.named_steps['preprocessor']
            fitted_model = full_pipeline.named_steps['model']
            X_processed = fitted_preprocessor.transform(X)
            feature_names = [f"feat_{i}" for i in range(X_processed.shape[1])]
            plots = generate_plots(fitted_model, X_processed, feature_names)
            if MLFLOW_AVAILABLE:
                for name, path in plots.items():
                    mlflow.log_artifact(path)
        except Exception as e:
            logger.warning(f"Plotting skipped: {e}")
                
        if not args.train_only:
            logger.info("Step 4: Generating Submission")
            preds_log = full_pipeline.predict(X_test)
            preds_final = np.expm1(preds_log)
            submission = pd.DataFrame({'Id': test_ids, 'SalePrice': preds_final})
            submission.to_csv("submission.csv", index=False)
            logger.info("Submission saved to submission.csv")

    logger.info("Run Complete.")

    print("\n" + "="*60)
    print("        DEPLOYMENT ARTIFACTS READY")
    print("="*60)
    print(f"\n[1] Model file created: model_pipeline.pkl")
    print(f"[2] To build Docker image:")
    print(f"    docker build -t ames-housing-api .")
    print("-" * 40)

if __name__ == "__main__":
    main()