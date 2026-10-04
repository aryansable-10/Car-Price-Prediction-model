"""
preprocess_train.py
────────────────────
Loads car_price_prediction.csv, cleans data, trains a
RandomForestRegressor and saves the model + metadata to model/.
Run once before launching the Streamlit app:
    python preprocess_train.py
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_PATH  = "car_price_prediction.csv"
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "car_price_model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "feature_meta.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# ── 1. Load ──────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df


# ── 2. Clean ─────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip numeric values embedded in strings (e.g. "74 bhp" → 74.0)
    def strip_unit(series: pd.Series) -> pd.Series:
        extracted: pd.Series = series.astype(str).str.extract(r"([\d.]+)", expand=False)
        return pd.to_numeric(extracted, errors="coerce").astype(float)

    df["mileage"]   = strip_unit(df["mileage(km/ltr/kg)"])
    df["engine_cc"] = strip_unit(df["engine"])
    df["max_power"] = strip_unit(df["max_power"])

    # Current age of car
    df["car_age"] = 2024 - df["year"]

    # Drop columns not needed for modelling
    df.drop(columns=["name", "year", "mileage(km/ltr/kg)", "engine"], inplace=True)

    # Drop rows where target is missing
    df.dropna(subset=["selling_price"], inplace=True)

    # Fill numeric NaNs with median
    for col in ["mileage", "engine_cc", "max_power", "seats"]:
        df[col] = df[col].fillna(df[col].median())

    print(f"After cleaning: {len(df):,} rows")
    return df


# ── 3. Encode ────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["fuel", "seller_type", "transmission", "owner"]

def encode_data(df: pd.DataFrame):
    """Label-encode categoricals; return (df_encoded, label_encoders, feature_cols)."""
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [c for c in df.columns if c != "selling_price"]
    return df, encoders, feature_cols


# ── 4. Train ─────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols]
    y = np.log1p(df["selling_price"])          # log-transform target → better RMSE

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
    r2  = r2_score(y_test, y_pred)

    print("\n-- Model Performance ------------------")
    print(f"  R2  (log scale) : {r2:.4f}")
    print(f"  MAE (INR)       : {mae:,.0f}")
    print("---------------------------------------\n")

    return model


# ── 5. Save ───────────────────────────────────────────────────────────────────
def save_artifacts(model, encoders, feature_cols, df_clean):
    meta = {
        "feature_cols": feature_cols,
        "encoders": encoders,
        "categorical_cols": CATEGORICAL_COLS,
        "category_values": {
            col: list(enc.classes_) for col, enc in encoders.items()
        },
        "numeric_ranges": {
            col: (float(df_clean[col].min()), float(df_clean[col].max()))
            for col in ["km_driven", "mileage", "engine_cc", "max_power", "seats", "car_age"]
        },
        "feature_importances": dict(
            zip(feature_cols, model.feature_importances_.tolist())
        ),
    }
    joblib.dump(model, MODEL_PATH)
    joblib.dump(meta,  META_PATH)
    print(f"Saved model  -> {MODEL_PATH}")
    print(f"Saved meta   -> {META_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_raw    = load_data(DATA_PATH)
    df_clean  = clean_data(df_raw)
    df_enc, encoders, feature_cols = encode_data(df_clean)
    model     = train(df_enc, feature_cols)
    save_artifacts(model, encoders, feature_cols, df_clean)
    print("Done. Now launch the app:  streamlit run app.py")
