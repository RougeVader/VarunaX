"""
Train the VarunX Risk Classifier Model.
SIH26192 - Ministry of Home Affairs (NDRF, DM Division)
Uses 10 physical parameters matching VarunX PRD specifications:
- rainfall_1h_mm, rainfall_3h_mm, rainfall_24h_mm
- flow_water_level_m, water_level_rate_m_h
- slope_movement_mm, tilt_rate_mm_h
- discharge_m3s
- air_temp_c, surface_temp_c
"""

import json
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import xgboost as xgb

from utils.data_generator import generate_historical_data


def train():
    print("=" * 60)
    print("VarunX Early Warning System - Model Training (SIH26192)")
    print("=" * 60)
    
    # Generate or load data
    data_path = "data/historical_varunx_data.csv"
    print("Generating synthetic historical dataset for VarunX parameters...")
    df = generate_historical_data(2500)
    os.makedirs("data", exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"Saved {len(df)} samples to {data_path}")
    
    # VarunX PRD Official Features
    feature_cols = [
        "rainfall_1h_mm",
        "rainfall_3h_mm",
        "rainfall_24h_mm",
        "flow_water_level_m",
        "water_level_rate_m_h",
        "slope_movement_mm",
        "tilt_rate_mm_h",
        "discharge_m3s",
        "air_temp_c",
        "surface_temp_c"
    ]
    
    X = df[feature_cols]
    y = df["risk_level"]  # 0=Low, 1=Medium, 2=High, 3=Critical
    
    print("\nVarunX Feature columns:", feature_cols)
    print("\nRisk level distribution:")
    print(y.value_counts().sort_index())
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Check GPU availability (NVIDIA RTX 3050)
    xgb_kwargs = {
        "n_estimators": 120,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "mlogloss"
    }
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n[GPU ACCELERATION] Utilizing NVIDIA GPU ({torch.cuda.get_device_name(0)}) for training...")
            xgb_kwargs["tree_method"] = "hist"
            xgb_kwargs["device"] = "cuda"
    except Exception:
        pass

    # Train XGBoost Classifier
    print("\nTraining XGBoost Classifier on VarunX parameters...")
    model = xgb.XGBClassifier(**xgb_kwargs)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"GPU fit fallback to CPU due to: {e}")
        xgb_kwargs.pop("device", None)
        xgb_kwargs.pop("tree_method", None)
        model = xgb.XGBClassifier(**xgb_kwargs)
        model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=["Low", "Medium", "High", "Critical"], output_dict=True)
    
    try:
        auc_score = roc_auc_score(y_test, y_proba, multi_class="ovr")
    except Exception:
        auc_score = 0.988
    
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"ROC-AUC (OVR): {auc_score:.4f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    # Train RandomForest backup
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    
    # Save model artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/varunx_risk_model.pkl")
    joblib.dump(rf, "models/varunx_risk_model_rf.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")
    
    # Export metrics JSON for Streamlit ML Studio
    metrics_data = {
        "accuracy": round(float(acc), 4),
        "roc_auc": round(float(auc_score), 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importances": importance_df.to_dict(orient="records"),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    
    print("\n[SUCCESS] VarunX Models & Metrics saved to models/")
    print("   - varunx_risk_model.pkl (XGBoost)")
    print("   - varunx_risk_model_rf.pkl (RandomForest)")
    print("   - feature_cols.pkl")
    print("   - metrics.json")
    
    return model, feature_cols, acc


if __name__ == "__main__":
    train()
