import os
import json
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Any, Optional

# Split data into train/validation/test sets based on publication IDs
def split_data(df: pd.DataFrame, manual_pub_ids: List[str], auto_pub_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Select 1 publication manually curated for test, 1 for validation
    # Do same for auto curated publications
    test_manual = manual_pub_ids[:1] if len(manual_pub_ids) >= 1 else []
    valid_manual = manual_pub_ids[1:2] if len(manual_pub_ids) >= 2 else []
    
    test_auto = auto_pub_ids[:1] if len(auto_pub_ids) >= 1 else []
    valid_auto = auto_pub_ids[1:2] if len(auto_pub_ids) >= 2 else []

    test_ids = test_manual + test_auto
    valid_ids = valid_manual + valid_auto
    train_ids = [x for x in df['pub_id'].unique() if x not in test_ids + valid_ids]

    df_train = df[df['pub_id'].isin(train_ids)].copy()
    df_val = df[df['pub_id'].isin(valid_ids)].copy()
    df_test = df[df['pub_id'].isin(test_ids)].copy()

    print(f"Split summary:")
    print(f"  Train: {len(df_train)} rows, {len(train_ids)} publications")
    print(f"  Valid: {len(df_val)} rows, {len(valid_ids)} publications")
    print(f"  Test:  {len(df_test)} rows, {len(test_ids)} publications")

    return df_train, df_val, df_test

# Train XGBoost Ranker model
from typing import List
import pandas as pd
import xgboost as xgb

def train_ranker(df_train: pd.DataFrame, df_val: pd.DataFrame, features: List[str], target: str = 'label', **xgb_params) -> xgb.XGBRanker:
    if df_train.empty:
        raise ValueError("df_train is empty")

    # Build safe query id to avoid collisions across papers
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_train["query_id"] = df_train["pub_id"].astype(str) + "::" + df_train["bib_ref_id"].astype(str)
    df_val["query_id"] = df_val["pub_id"].astype(str) + "::" + df_val["bib_ref_id"].astype(str)

    # Sort by query id so groups are contiguous
    df_train_sorted = df_train.sort_values(["query_id"]).reset_index(drop=True)
    X_train = df_train_sorted[features]
    y_train = df_train_sorted[target]
    group_train = df_train_sorted.groupby("query_id", sort=False).size().tolist()

    model = xgb.XGBRanker(
        objective='rank:pairwise',
        learning_rate=0.1,
        n_estimators=1000,
        random_state=42,
        early_stopping_rounds=10,
        **xgb_params
    )

    if df_val.empty:
        print("Warning: df_val is empty, training without validation")
        model.fit(X_train, y_train, group=group_train, verbose=False)
    else:
        df_val_sorted = df_val.sort_values(["query_id"]).reset_index(drop=True)
        X_val = df_val_sorted[features]
        y_val = df_val_sorted[target]
        group_val = df_val_sorted.groupby("query_id", sort=False).size().tolist()

        model.fit(
            X_train, y_train,
            group=group_train,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val],
            verbose=False
        )

    print("Best iteration:", getattr(model, "best_iteration", None))
    print("Best score:", getattr(model, "best_score", None))
    return model


# Predict rankings for each bib_ref_id
def predict_rankings(model, df: pd.DataFrame, features: List[str], top_k: int = 5) -> Dict[str, List[str]]:
    df_pred = df.copy()
    
    df_pred['score'] = model.predict(df_pred[features])
    
    # Rank and select top K candidates per bib_ref_id
    result = {}
    for bib_ref_id, group in df_pred.groupby('bib_ref_id'):
        group_sorted = group.sort_values('score', ascending=False)
        top_candidates = group_sorted['candidate_arxiv_id'].tolist()[:top_k]
        result[bib_ref_id] = top_candidates
    
    return result

# Export predictions to JSON file
def export_predictions( predictions: Dict[str, List[str]], groundtruth: Optional[Dict[str, str]] = None, out_path: str = '.', partition: str = 'test', filename: str = 'pred.json') -> str:
    pred_json = {
        "partition": partition,
        "groundtruth": groundtruth or {},
        "prediction": predictions
    }
    
    os.makedirs(out_path, exist_ok=True)
    output_file = os.path.join(out_path, filename)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pred_json, f, indent=2, ensure_ascii=False)
    
    print(f"Predictions saved to: {output_file}")
    print(f"  Total predictions: {len(predictions)}")
    
    return output_file

# Save model to file
def save_model(model, path: str, filename: str = 'model.joblib') -> str:
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")
    return filepath

# Load model from file
def load_model(filepath: str):
    """Load model"""
    return joblib.load(filepath)

# Get feature importance from model
def get_feature_importance(model, features: List[str]) -> pd.DataFrame:
    importance = model.feature_importances_
    df_imp = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    return df_imp