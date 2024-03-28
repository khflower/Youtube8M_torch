import torch

def calculate_hit_at_one(predictions, actuals):
    """
    Performs a PyTorch calculation of the hit at one.

    Args:
        predictions: Tensor containing the outputs of the model.
            Dimensions are 'batch' x 'num_classes'.
        actuals: Tensor containing the ground truth labels.
            Dimensions are 'batch' x 'num_classes'.

    Returns:
        float: The average hit at one across the entire batch.
    """
    _, top_prediction = predictions.topk(1, dim=1)  # Get the top prediction
    top_prediction = top_prediction.squeeze(1)  # Remove the extra dimension

    hits = actuals.gather(1, top_prediction.unsqueeze(1)).squeeze(1)  # Gather the actual values for top predictions
    hit_at_one = hits.float().mean().item()  # Calculate the mean hit@1
    return hit_at_one