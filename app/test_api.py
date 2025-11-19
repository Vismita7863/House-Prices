import requests
import json

API_URL = 'http://127.0.0.1:5001/predict'

sample_house = {
    "MSSubClass": 20,
    "MSZoning": "RH",
    "LotFrontage": 81.0,
    "LotArea": 14267,
    "Street": "Pave",
    "Alley": "None",
    "LotShape": "IR1",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Corner",
    "LandSlope": "Gtl",
    "Neighborhood": "NAmes",
    "Condition1": "Norm",
    "Condition2": "Norm",
    "BldgType": "1Fam",
    "HouseStyle": "1Story",
    "OverallQual": 6,
    "OverallCond": 6,
    "YearBuilt": 1961,
    "YearRemodAdd": 1961,
    "RoofStyle": "Hip",
    "RoofMatl": "CompShg",
    "Exterior1st": "Wd Sdng",
    "Exterior2nd": "Wd Sdng",
    "MasVnrType": "BrkFace",
    "MasVnrArea": 108.0,
    "ExterQual": "TA",
    "ExterCond": "TA",
    "Foundation": "CBlock",
    "BsmtQual": "TA",
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "ALQ",
    "BsmtFinSF1": 923.0,
    "BsmtFinType2": "Unf",
    "BsmtFinSF2": 0.0,
    "BsmtUnfSF": 406.0,
    "TotalBsmtSF": 1329.0,
    "Heating": "GasA",
    "HeatingQC": "TA",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "1stFlrSF": 1329,
    "2ndFlrSF": 0,
    "LowQualFinSF": 0,
    "GrLivArea": 1329,
    "BsmtFullBath": 0.0,
    "BsmtHalfBath": 0.0,
    "FullBath": 1,
    "HalfBath": 1,
    "BedroomAbvGr": 3,
    "KitchenAbvGr": 1,
    "KitchenQual": "Gd",
    "TotRmsAbvGrd": 6,
    "Functional": "Typ",
    "Fireplaces": 0,
    "FireplaceQu": "None",
    "GarageType": "Attchd",
    "GarageYrBlt": 1961.0,
    "GarageFinish": "Unf",
    "GarageCars": 1.0,
    "GarageArea": 312.0,
    "GarageQual": "TA",
    "GarageCond": "TA",
    "PavedDrive": "Y",
    "WoodDeckSF": 393,
    "OpenPorchSF": 36,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    "PoolArea": 0,
    "PoolQC": "None",
    "Fence": "None",
    "MiscFeature": "Gar2",
    "MiscVal": 12500,
    "MoSold": 6,
    "YrSold": 2010,
    "SaleType": "WD",          
    "SaleCondition": "Normal"  
}

try:
    print(f"Sending request to {API_URL}...")
    response = requests.post(API_URL, json=sample_house) 
    
    response.raise_for_status()

    print("\n API Response (Success):\n")
    print(json.dumps(response.json(), indent=4))

except requests.exceptions.RequestException as e:
    print(f"\n ERROR connecting to API: {e}")
    if e.response:
        print("Server returned:", e.response.text)