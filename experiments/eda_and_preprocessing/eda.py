import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats

sns.set_style("whitegrid")

df_train = pd.read_csv('train.csv')

plt.figure(figsize=(12, 6))
sns.histplot(df_train['SalePrice'], kde=True, bins=50)
plt.title('Distribution of SalePrice', fontsize=15)
plt.xlabel('Sale Price ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

df_train['SalePrice'] = np.log1p(df_train['SalePrice'])

plt.figure(figsize=(12, 6))
sns.histplot(df_train['SalePrice'], kde=True, bins=50, color='green')
plt.title('Log-Transformed Distribution of SalePrice', fontsize=15)
plt.xlabel('Log of Sale Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

missing_values = df_train.isnull().sum().sort_values(ascending=False)
missing_percent = (df_train.isnull().sum() / len(df_train)).sort_values(ascending=False)
missing_data = pd.concat([missing_values, missing_percent], axis=1, keys=['Total', 'Percent'])
print("Features with missing values:\n")
print(missing_data[missing_data['Total'] > 0])

numerical_features = df_train.select_dtypes(include=np.number)
categorical_features = df_train.select_dtypes(exclude=np.number)
print(f"\nTotal Features: {df_train.shape[1]}")
print(f"Numerical features: {len(numerical_features.columns)}")
print(f"Categorical features: {len(categorical_features.columns)}")

plt.figure(figsize=(14, 12))

correlation_matrix = df_train.select_dtypes(include=np.number).corr()

top_corr_features = correlation_matrix.nlargest(15, 'SalePrice')['SalePrice'].index
top_corr_matrix = df_train[top_corr_features].corr() 

sns.heatmap(top_corr_matrix, annot=True, cmap='viridis', fmt='.2f')
plt.title('Top 15 Features Correlated with SalePrice', fontsize=16)
plt.show()

plt.figure(figsize=(15, 8))
sns.boxplot(x=df_train['OverallQual'], y=df_train['SalePrice'])
plt.title('SalePrice vs. Overall Quality', fontsize=15)
plt.xlabel('Overall Quality', fontsize=12)
plt.ylabel('Log of Sale Price', fontsize=12)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_train['GrLivArea'], y=df_train['SalePrice'])
plt.title('SalePrice vs. Ground Living Area', fontsize=15)
plt.xlabel('GrLivArea (sq. ft)', fontsize=12)
plt.ylabel('Log of Sale Price', fontsize=12)
plt.axvline(x=4500, color='r', linestyle='--') 
plt.show()
