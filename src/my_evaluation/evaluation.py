from typing import Dict, List


def calculate_mrr(
    predictions: Dict[str, List[str]], 
    groundtruth: Dict[str, str], 
    top_k: int = 5
) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) for a set of predictions.

    The MRR is the average of the reciprocal ranks of the first correct answer 
    for a set of queries.

    Parameters
    ----------
    predictions : Dict[str, List[str]]
        Dictionary mapping query keys (e.g., BibTeX keys) to a list of 
        predicted candidate IDs, ranked by relevance.
    groundtruth : Dict[str, str]
        Dictionary mapping query keys to the single correct ground truth ID.
    top_k : int, default=5
        Only consider the top K predictions for ranking.

    Returns
    -------
    float
        The calculated MRR score rounded to 4 decimal places. Returns 0.0 if 
        groundtruth is empty.
    """
    if not groundtruth:
        return 0.0
    
    mrr_total = 0.0
    count = 0
    
    for bib_key, true_id in groundtruth.items():
        # Retrieve top K predictions for the specific query
        preds = predictions.get(bib_key, [])[:top_k]
        count += 1
        
        # Find the rank of the true ID within the predictions
        rank = 0
        for i, pred_id in enumerate(preds, 1):
            if pred_id == true_id:
                rank = i
                break
        
        # Add reciprocal rank (1/rank) to total if match found
        if rank > 0:
            mrr_total += 1.0 / rank
    
    return round(mrr_total / count, 4) if count > 0 else 0.0
