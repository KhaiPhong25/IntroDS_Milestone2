import ast
import pandas as pd
import numpy as np
import Levenshtein
from typing import List, Tuple, Any

def safe_parse(val: Any) -> List[Any]:
    """
    Safely parse a string or list representation into a list object.

    Handles standard lists, pandas NaNs, string representations of lists 
    (e.g., "['a', 'b']"), and comma-separated strings.

    Parameters
    ----------
    val : Any
        Input value (str, list, or NaN).

    Returns
    -------
    List[Any]
        Parsed list. Returns an empty list if input is NaN or empty.
    """
    if isinstance(val, list):
        return val
    
    # Handle NaN (pandas) or empty/None values
    if pd.isna(val) or not val:
        return []
    
    if isinstance(val, str):
        try:
            # Attempt to evaluate string as a Python literal list
            return ast.literal_eval(val)
        except:
            # Fallback: Split by comma for simple CSV-like strings
            return [x.strip() for x in val.split(",") if x.strip()]
            
    return []


def calculate_text_features(title1: str, title2: str, threshold: float = 0.8) -> Tuple[float, int]:
    """
    Calculate text similarity features: Soft Jaccard score and Length Difference.

    Soft Jaccard uses Levenshtein ratio to match tokens fuzzily rather than
    requiring exact string equality.

    Parameters
    ----------
    title1, title2 : str
        Input title strings to compare.
    threshold : float, default=0.8
        Minimum Levenshtein ratio required to consider two tokens a match.

    Returns
    -------
    Tuple[float, int]
        - Soft Jaccard similarity score (0.0 to 1.0).
        - Absolute difference in character length.
    """
    if not title1 or not title2:
        return 0, 0
    
    if not isinstance(title1, str) or not isinstance(title2, str):
        return 0, 0
        
    # 1. Calculate Length Difference
    len_diff = abs(len(title1) - len(title2))
    
    # 2. Tokenize inputs
    tokens1 = title1.split()
    tokens2 = title2.split()

    # 3. Calculate Soft Jaccard Similarity
    # Strategy: Find best fuzzy match for each token in title1 against title2
    match_count = 0
    used_indices = set() # Track matched indices in title2 to prevent double-counting
    
    for t1 in tokens1:
        best_score = 0.0
        best_idx = -1
        
        for i, t2 in enumerate(tokens2):
            if i in used_indices:
                continue
            
            # Compute token-level similarity
            score = Levenshtein.ratio(t1, t2)
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        # Increment match count if the best match meets the threshold
        if best_score > threshold:
            match_count += 1
            if best_idx >= 0:
                used_indices.add(best_idx)
    
    # Formula: J = Intersection / Union
    # Union = len(A) + len(B) - Intersection
    total_len = len(tokens1) + len(tokens2)
    soft_jaccard = match_count / (total_len - match_count) if (total_len - match_count) > 0 else 0.0

    return soft_jaccard, len_diff


def calculate_author_features(
    short_list1: Any, 
    short_list2: Any, 
    tokens_list1: Any, 
    tokens_list2: Any
) -> Tuple[float, float]:
    """
    Calculate author similarity features: Overlap Score and Levenshtein Ratio.

    Parameters
    ----------
    short_list1, short_list2 : Any
        Lists (or string representations) of author short names (e.g., "Doe J").
    tokens_list1, tokens_list2 : Any
        Lists (or string representations) of author name tokens for detailed comparison.

    Returns
    -------
    Tuple[float, float]
        - Overlap Score: Jaccard similarity of exact short name matches.
        - Levenshtein Ratio: Average best-match score between author tokens.
    """
    # 1. Parse inputs into lists
    short_list1 = safe_parse(short_list1)
    short_list2 = safe_parse(short_list2)
    tokens_list1 = safe_parse(tokens_list1)
    tokens_list2 = safe_parse(tokens_list2)
    
    # 2. Author Overlap Score (Jaccard on Short Forms)
    set1 = set(short_list1)
    set2 = set(short_list2)
    
    if len(set1) == 0 and len(set2) == 0:
        overlap_score = 0.0
    else:
        inter = len(set1.intersection(set2))
        union = len(set1.union(set2))
        overlap_score = inter / union if union > 0 else 0.0

    # 3. Author Levenshtein Ratio (Mean of Best Matches)
    if len(tokens_list1) == 0 or len(tokens_list2) == 0:
        lev_ratio = 0.0
    else:
        best_scores = [] 
        used_indices = set() # Track indices in list2 to prevent reuse
        
        # Cross-compare each author in list1 with list2
        for author1_tokens in tokens_list1:
            str1 = " ".join(author1_tokens)
            
            best_score = 0.0
            best_idx = -1
            
            for i, author2_tokens in enumerate(tokens_list2):
                if i in used_indices:
                    continue

                str2 = " ".join(author2_tokens)
                
                # Calculate Levenshtein Ratio for this pair
                score = Levenshtein.ratio(str1, str2)
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            best_scores.append(best_score)
            if best_idx >= 0:
                used_indices.add(best_idx)
        
        # Compute mean score across all authors in list1
        lev_ratio = sum(best_scores) / len(best_scores) if best_scores else 0.0

    return overlap_score, lev_ratio


def calculate_year_features(year1: Any, year2: Any) -> float:
    """
    Calculate the absolute difference between two years.

    Parameters
    ----------
    year1, year2 : Any
        Year values (can be string, int, float, or None).

    Returns
    -------
    float
        Absolute difference between years, or -1 if invalid/missing.
    """
    try:
        y1 = float(year1) if year1 is not None else np.nan
        y2 = float(year2) if year2 is not None else np.nan
        
        # Return -1 if either year is missing
        if np.isnan(y1) or np.isnan(y2):
            return -1
        
        return abs(y1 - y2)
    except (ValueError, TypeError):
        # Fallback for parsing errors
        return -1


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate similarity features for BibTeX-arXiv candidate pairs.

    Computes features based on Title, Author, and Year comparisons to be used
    in the ranking model.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing candidate pairs and their raw metadata columns
        (e.g., 'bib_title_clean', 'candidate_title_clean').

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the generated features and necessary metadata keys.
    """
    df_feat = df.copy()

    # 1. TITLE FEATURES: Soft Jaccard & Length Difference
    # Uses zip(*) to unpack the tuple return values from apply
    df_feat["Title_Soft_Jaccard"], df_feat["Title_Length_Diff"] = zip(*df_feat.apply(
        lambda r: calculate_text_features(
            r.get("bib_title_clean", ""), 
            r.get("candidate_title_clean", "")
        ), axis=1
    ))

    # 2. AUTHOR FEATURES: Overlap Score & Levenshtein Ratio
    df_feat["Author_Overlap_Score"], df_feat["Author_Levenshtein_Ratio"] = zip(*df_feat.apply(
        lambda r: calculate_author_features(
            r.get("bib_authors_clean", []),
            r.get("candidate_authors_clean", []),
            r.get("bib_author_tokens", []),
            r.get("candidate_author_tokens", [])
        ), axis=1
    ))

    # 3. YEAR FEATURES: Absolute Year Difference
    df_feat["Year_Diff"] = df_feat.apply(
        lambda r: calculate_year_features(
            r.get("bib_year", ""), 
            r.get("candidate_year", "")
        ), axis=1
    )
    
    # 4. Feature Selection
    feature_cols = [
        # Metadata
        "label",
        "pub_id",
        "bib_key",
        "candidate_arxiv_id",
        "source",
        # Title features
        "Title_Soft_Jaccard",
        "Title_Length_Diff",
        # Author features
        "Author_Overlap_Score",
        "Author_Levenshtein_Ratio",
        # Year features
        "Year_Diff"
    ]
    
    # Ensure we only select columns that actually exist (safeguard)
    existing_cols = [c for c in feature_cols if c in df_feat.columns]
    
    return df_feat[existing_cols]
