import joblib
import pandas as pd

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")

NUMERIC_COLS = [
    'age', 'campaign', 'pdays', 'previous',
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
    'euribor3m', 'nr.employed'
]

CALL_COST = 5
REVENUE = 100

def preprocess_input(df):
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

    for col in NUMERIC_COLS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded[NUMERIC_COLS] = scaler.transform(df_encoded[NUMERIC_COLS])
    return df_encoded

def predict_customer(df):
    processed = preprocess_input(df)
    prob = model.predict_proba(processed)[0][1]

    expected_profit = prob * REVENUE - CALL_COST

    if expected_profit > 0:
        action = "CALL"
    else:
        action = "SKIP"

    return {
        "probability": prob,
        "expected_profit": expected_profit,
        "action": action
    }