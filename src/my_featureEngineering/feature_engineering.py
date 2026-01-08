import ast
import pandas as pd
import numpy as np
import Levenshtein

def safe_parse(val):
    if isinstance(val, list):
        return val
    if pd.isna(val) or not val:
        return []
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except:
            return [x.strip() for x in val.split(",") if x.strip()]
    return []

def calculate_text_features(title1, title2, threshold=0.8):
    if not title1 or not title2:
        return 0, 0
    
    if not isinstance(title1, str) or not isinstance(title2, str):
        return 0, 0
    # Length Diff
    len_diff = abs(len(title1) - len(title2))
    
    # Prepare tokens for Jaccard
    tokens1 = title1.split()
    tokens2 = title2.split()

    # Soft Jaccard (Combined with Levenshtein Ratio per token)
    # Idea: For each token in title1, find the best matching token in title2
    match_count = 0
    used_indices = set() # Mark tokens in title2 that are already matched
    
    for t1 in tokens1:
        best_score = 0.0
        best_idx = -1
        
        for i, t2 in enumerate(tokens2):
            if i in used_indices:
                continue
            
            # Calculate Levenshtein Ratio
            score = Levenshtein.ratio(t1, t2)
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        # If best match exceeds threshold, count as a match
        if best_score > threshold:
            match_count += 1
            if best_idx >= 0:
                used_indices.add(best_idx)
    
    # Soft Jaccard: Match / (Len1 + Len2 - Match)
    total_len = len(tokens1) + len(tokens2)
    soft_jaccard = match_count / (total_len - match_count) if (total_len - match_count) > 0 else 0.0

    return soft_jaccard, len_diff

# Calculate author features
def calculate_author_features(short_list1, short_list2, tokens_list1, tokens_list2):
    # Author Overlap Score (Based on Short Forms - "LastName FirstInitial")
    # Idea: Compare two sets of short forms for overlap
    short_list1 = safe_parse(short_list1)
    short_list2 = safe_parse(short_list2)
    tokens_list1 = safe_parse(tokens_list1)
    tokens_list2 = safe_parse(tokens_list2)
    set1 = set(short_list1)
    set2 = set(short_list2)
    
    if len(set1) == 0 and len(set2) == 0:
        overlap_score = 0.0
    else:
        inter = len(set1.intersection(set2))
        union = len(set1.union(set2))
        overlap_score = inter / union if union > 0 else 0.0

    # Author Levenshtein Ratio (Mean of Best Matches)
    # Idea: For each author in list1, find best matching author in list2 based on Levenshtein ratio of their name tokens
    if len(tokens_list1) == 0 or len(tokens_list2) == 0:
        lev_ratio = 0.0
    else:
        best_scores = [] # Save best scores for each author in list1
        
        used_indices = set()
        # Cross-compare each author in list1 with all authors in list2 to find best match author
        for author1_tokens in tokens_list1:
            # Join tokens to form a single string for comparison
            str1 = " ".join(author1_tokens)
            
            # Find best matching author in list2
            best_score = 0.0
            best_idx = -1
            for i, author2_tokens in enumerate(tokens_list2):
                if i in used_indices:
                    continue

                str2 = " ".join(author2_tokens)
                
                # Calculate Levenshtein Ratio
                score = Levenshtein.ratio(str1, str2)
                
                # Update best score
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            # Save best score for this author
            best_scores.append(best_score)
            if best_idx >= 0:
                used_indices.add(best_idx)
        
        # Calculate mean of best scores
        lev_ratio = sum(best_scores) / len(best_scores) if best_scores else 0.0

    return overlap_score, lev_ratio

# Calculate year features
def calculate_year_features(year1, year2):
    # Handle missing values
    try:
        y1 = float(year1) if year1 is not None else np.nan
        y2 = float(year2) if year2 is not None else np.nan
        
        # Check for NaN
        if np.isnan(y1) or np.isnan(y2):
            return -1
        
        return abs(y1 - y2)
    except (ValueError, TypeError):
        # If conversion fails, return -1
        return -1


# Main feature engineering function
def feature_engineering(df):
    df_feat = df.copy()

    # TITLE FEATURES
    df_feat["Title_Soft_Jaccard"], df_feat["Title_Length_Diff"] = zip(*df_feat.apply(
        lambda r: calculate_text_features(
            r.get("bib_title_clean", ""), 
            r.get("candidate_title_clean", "")
        ), axis=1
    ))

    # AUTHOR FEATURES
    df_feat["Author_Overlap_Score"], df_feat["Author_Levenshtein_Ratio"] = zip(*df_feat.apply(
        lambda r: calculate_author_features(
            r.get("bib_authors_clean", []),
            r.get("candidate_authors_clean", []),
            r.get("bib_author_tokens", []),
            r.get("candidate_author_tokens", [])
        ), axis=1
    ))

    # YEAR FEATURES
    df_feat["Year_Diff"] = df_feat.apply(
        lambda r: calculate_year_features(
            r.get("bib_year", ""), 
            r.get("candidate_year", "")
        ), axis=1
    )
    
    # SELECT FEATURE COLUMNS
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
    
    existing_cols = [c for c in feature_cols if c in df_feat.columns]
    return df_feat[existing_cols]
