from typing import List, Set


def _calculate_precision(predicted_ids: Set[str], ground_truth_ids: Set[str]) -> float:
    """calculate precision - how many are truly relevant"""
    if not predicted_ids or not ground_truth_ids:
        return 0.0
    
    true_positives = len(predicted_ids & ground_truth_ids)
    precision = true_positives / len(predicted_ids) if predicted_ids else 0.0
    
    return precision


def calculate_all_metrics(predicted_ids: List[str], ground_truth_ids: List[str]) -> dict:
    """calculate all metrics - simplified version, only return precision"""
    
    # convert to set for efficient calculation
    predicted_set = set(predicted_ids)
    ground_truth_set = set(ground_truth_ids)
    
    precision = _calculate_precision(predicted_set, ground_truth_set)
    
    # use precision as the only evaluation metric
    return {
        "eval_score": precision,  # comprehensive score is precision
        "precision": precision,   # precision
    }
