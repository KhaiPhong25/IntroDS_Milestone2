import os
import json
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Any, Optional

# Negative sampling per query to handle imbalanced data
def sample_negatives_per_query(df: pd.DataFrame, features: List[str], target: str = 'label', k_negatives: int = 15, hard_ratio: float = 0.6, random_seed: int = 42) -> pd.DataFrame:
    """
    Apply negative sampling per query to reduce data imbalance.
    
    For each query:
    - Keep all positive samples (label=1)
    - Sample K negatives:
        * hard_ratio * K: hard negatives (most similar to query but incorrect)
        * (1 - hard_ratio) * K: random negatives
    
    Args:
        df: DataFrame with columns [query_id, label, ...features]
        features: List of feature column names to use for computing similarity
        target: Target column name (default='label')
        k_negatives: Number of negatives to sample per query (default=15)
        hard_ratio: Ratio of hard negatives (default=0.6 means 60% hard, 40% random)
        random_seed: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame with balanced negatives per query
    """
    np.random.seed(random_seed)
    
    if 'query_id' not in df.columns:
        raise ValueError("DataFrame must have 'query_id' column")
    
    sampled_dfs = []
    
    for query_id, group in df.groupby('query_id'):
        # Separate positives and negatives
        positives = group[group[target] == 1]
        negatives = group[group[target] == 0]
        
        # Keep all positives
        sampled_dfs.append(positives)
        
        if len(negatives) == 0:
            continue
        
        # If negatives <= k_negatives, keep all
        if len(negatives) <= k_negatives:
            sampled_dfs.append(negatives)
            continue
        
        # Calculate how many hard and random negatives to sample
        n_hard = int(k_negatives * hard_ratio)
        n_random = k_negatives - n_hard
        
        # Get hard negatives: negatives with highest feature similarity to positive
        if len(positives) > 0 and n_hard > 0:
            # Use features to compute similarity
            try:
                # Get feature vectors
                pos_features = positives[features].values
                neg_features = negatives[features].values
                
                # Compute average positive feature vector
                pos_mean = pos_features.mean(axis=0).reshape(1, -1)
                
                # Compute similarity between each negative and positive
                similarities = cosine_similarity(neg_features, pos_mean).flatten()
                
                # Get indices of top N similar negatives (hard negatives)
                hard_indices = np.argsort(similarities)[-n_hard:]
                hard_negatives = negatives.iloc[hard_indices]
                
                # Get remaining negatives for random sampling
                remaining_mask = np.ones(len(negatives), dtype=bool)
                remaining_mask[hard_indices] = False
                remaining_negatives = negatives[remaining_mask]
                
            except Exception as e:
                # Fallback: if similarity computation fails, do pure random sampling
                print(f"Warning: Similarity computation failed for query {query_id}: {e}")
                hard_negatives = pd.DataFrame()
                remaining_negatives = negatives
        else:
            hard_negatives = pd.DataFrame()
            remaining_negatives = negatives
        
        # Add hard negatives
        if len(hard_negatives) > 0:
            sampled_dfs.append(hard_negatives)
        
        # Sample random negatives from remaining
        if n_random > 0 and len(remaining_negatives) > 0:
            n_to_sample = min(n_random, len(remaining_negatives))
            random_negatives = remaining_negatives.sample(n=n_to_sample, random_state=random_seed)
            sampled_dfs.append(random_negatives)
    
    # Combine all sampled data
    df_sampled = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\nNegative Sampling Summary:")
    print(f"  Original: {len(df)} rows")
    print(f"  Sampled:  {len(df_sampled)} rows")
    print(f"  Reduction: {100 * (1 - len(df_sampled)/len(df)):.1f}%")
    print(f"  Positives: {len(df_sampled[df_sampled[target]==1])}")
    print(f"  Negatives: {len(df_sampled[df_sampled[target]==0])}")
    print(f"  Ratio (pos:neg): 1:{len(df_sampled[df_sampled[target]==0])/max(1, len(df_sampled[df_sampled[target]==1])):.1f}")
    
    return df_sampled

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

def train_ranker(df_train: pd.DataFrame, df_val: pd.DataFrame, features: List[str], target: str = 'label', use_negative_sampling: bool = True, k_negatives: int = 15, hard_ratio: float = 0.6, **xgb_params) -> xgb.XGBRanker:
    """
    Train XGBoost Ranker with optional negative sampling.
    
    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame
        features: List of feature columns
        target: Target column (default='label')
        use_negative_sampling: Whether to apply negative sampling (default=True)
        k_negatives: Number of negatives per query when sampling (default=15)
        hard_ratio: Ratio of hard negatives (default=0.6)
        **xgb_params: Additional XGBoost parameters
    
    Returns:
        Trained XGBRanker model
    """
    if df_train.empty:
        raise ValueError("df_train is empty")

    # Build safe query id to avoid collisions across papers
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_train["query_id"] = df_train["pub_id"].astype(str) + "::" + df_train["bib_key"].astype(str)
    df_val["query_id"] = df_val["pub_id"].astype(str) + "::" + df_val["bib_key"].astype(str)
    
    # Apply negative sampling if enabled
    if use_negative_sampling:
        print("\nApplying negative sampling to training data...")
        df_train = sample_negatives_per_query(
            df_train,
            features=features,
            target=target,
            k_negatives=k_negatives,
            hard_ratio=hard_ratio
        )

    # Sort by query id so groups are contiguous
    df_train_sorted = df_train.sort_values(["query_id"]).reset_index(drop=True)
    X_train = df_train_sorted[features]
    y_train = df_train_sorted[target]
    group_train = df_train_sorted.groupby("query_id", sort=False).size().tolist()

    model = xgb.XGBRanker(
        objective='rank:pairwise',
        learning_rate=0.05,
        n_estimators=1000,
        random_state=42,
        early_stopping_rounds=20,
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


# Predict rankings for each bib_key
def predict_rankings(model, df: pd.DataFrame, features: List[str], top_k: int = 5) -> Dict[str, Dict[str, List[str]]]:
    """
    Predict rankings for all candidates and return top-K per query.
    
    NOTE: This function scores ALL candidates in df, not just the sampled ones used in training.
    This ensures proper evaluation even when training used negative sampling.
    
    Args:
        model: Trained XGBRanker model
        df: DataFrame with ALL candidates (not sampled)
        features: List of feature columns
        top_k: Number of top candidates to return (default=5)
    
    Returns:
        Nested dictionary: {pub_id: {bib_key: [top-K arxiv_ids]}}
        This structure prevents key collisions when multiple publications have the same bib_key.
    """
    df_pred = df.copy()
    
    df_pred['score'] = model.predict(df_pred[features])
    
    # Rank and select top K candidates per query (pub_id, bib_key)
    result = {}
    for (pub_id, bib_key), group in df_pred.groupby(['pub_id', 'bib_key']):
        # Sort by score and select top K
        group_sorted = group.sort_values('score', ascending=False)
        top_candidates = group_sorted['candidate_arxiv_id'].tolist()[:top_k]
        
        # Build nested structure: {pub_id: {bib_key: [candidates]}}
        if pub_id not in result:
            result[pub_id] = {}
        result[pub_id][bib_key] = top_candidates
    
    return result

# Extract groundtruth (nested dict: {pub_id: {bib_key: arxiv_id}})
def extract_groundtruth(df):
    """Extract groundtruth as nested dict by pub_id"""
    groundtruth = {}
    for _, row in df[df['label'] == 1].iterrows():
        pub_id = row['pub_id']
        bib_key = row['bib_key']
        arxiv_id = row['candidate_arxiv_id']
        
        if pub_id not in groundtruth:
            groundtruth[pub_id] = {}
        groundtruth[pub_id][bib_key] = arxiv_id
    
    return groundtruth

def flatten_nested_dict(nested_dict):
    """
    Flatten nested dict {pub_id: {bib_key: [candidates]}} 
    to {bib_key: [candidates]} for evaluation.
    
    Note: If multiple publications have same bib_key, uses composite key "pub_id::bib_key"
    """
    flat = {}
    for pub_id, bib_predictions in nested_dict.items():
        for bib_key, candidates in bib_predictions.items():
            # Use composite key to avoid collisions
            composite_key = f"{pub_id}::{bib_key}"
            flat[composite_key] = candidates
    return flat

def flatten_groundtruth(nested_groundtruth):
    """Flatten nested groundtruth dict for evaluation"""
    flat = {}
    for pub_id, bib_truth in nested_groundtruth.items():
        for bib_key, arxiv_id in bib_truth.items():
            # Use composite key to match predictions
            composite_key = f"{pub_id}::{bib_key}"
            flat[composite_key] = arxiv_id
    return flat

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