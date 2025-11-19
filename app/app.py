# app.py
import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "model_pipeline.pkl"

model_pipeline = None

def load_model():
    global model_pipeline
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"SUCCESS: Pipeline loaded from {MODEL_PATH}")
    else:
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("PLEASE RUN 'python train_pipeline.py' FIRST to generate the model.")
        model_pipeline = None

def apply_feature_engineering(df):
    df = df.copy()
    
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
    
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    total_bsmt = df.get('TotalBsmtSF', 0)
    first_flr = df.get('1stFlrSF', 0)
    second_flr = df.get('2ndFlrSF', 0)
    df['TotalSF'] = total_bsmt + first_flr + second_flr
    
    if 'YearBuilt' in df.columns and 'YrSold' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    else:
        df['HouseAge'] = 0

    return df

@app.route('/predict', methods=['POST'])
def predict():
    if not model_pipeline:
        return jsonify({"error": "Model is not loaded."}), 500

    try:
        json_data = request.get_json()
        
        if isinstance(json_data, list):
            input_df = pd.DataFrame(json_data)
        else:
            input_df = pd.DataFrame([json_data])

        processed_df = apply_feature_engineering(input_df)
        log_predictions = model_pipeline.predict(processed_df)
        actual_predictions = np.expm1(log_predictions)

        response_data = {'predictions': [round(float(p), 2) for p in actual_predictions]}
        if len(actual_predictions) == 1:
            response_data['predicted_price'] = response_data['predictions'][0]

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)