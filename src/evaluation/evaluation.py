from typing import Dict, List


def calculate_mrr(predictions: Dict[str, List[str]],  groundtruth: Dict[str, str], top_k: int = 5) -> float:
    if not groundtruth:
        return 0.0
    
    mrr_total = 0.0
    count = 0
    
    for bib_key, true_id in groundtruth.items():
        preds = predictions.get(bib_key, [])[:top_k]
        count += 1
        
        # Find rank of true_id in preds
        rank = 0
        for i, pred_id in enumerate(preds, 1):
            if pred_id == true_id:
                rank = i
                break
        
        if rank > 0:
            mrr_total += 1.0 / rank
    
    return round(mrr_total / count, 4) if count > 0 else 0.0
