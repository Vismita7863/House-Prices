import pandas as pd
import numpy as np

def load_data(train_path='data/train.csv', test_path='data/test.csv'):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess(train_df, test_df):
    train_id = train_df['Id']
    test_id = test_df['Id']
    train_df.drop('Id', axis=1, inplace=True)
    test_df.drop('Id', axis=1, inplace=True)
    
    train_df = train_df.drop(
        train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)].index
    )
    
    train_df["SalePrice"] = np.log1p(train_df["SalePrice"])
    
    y_train = train_df.SalePrice.values
    all_data = pd.concat((train_df.drop(['SalePrice'], axis=1), test_df), ignore_index=True)

    for col in ('Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType'):
        all_data[col] = all_data[col].fillna('None')

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea', 'BsmtFinSF1',
                'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        all_data[col] = all_data[col].fillna(0)

    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 
                'Exterior2nd', 'SaleType', 'Functional', 'Utilities'):
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

    all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)

    all_data = pd.get_dummies(all_data, drop_first=True)

    X_train = all_data[:len(y_train)]
    X_test = all_data[len(y_train):]
    
    train_final = X_train.copy()
    train_final['SalePrice'] = y_train
    
    test_final = X_test.copy()
    
    return train_final, test_final, train_id, test_id

if __name__ == '__main__':
    train, test = load_data()
    train_cleaned, test_cleaned, _, _ = preprocess(train, test)

    train_cleaned.to_csv('data/train_cleaned.csv', index=False)
    test_cleaned.to_csv('data/test_cleaned.csv', index=False)

    print("✅ Data cleaning and preprocessing complete.")
    print(f"Cleaned training data saved to 'data/train_cleaned.csv' with shape: {train_cleaned.shape}")
    print(f"Cleaned test data saved to 'data/test_cleaned.csv' with shape: {test_cleaned.shape}")