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



def calculate_precision_at_equal_recall_rate_torch(predictions, actuals):
    """Performs a local (torch) calculation of the PERR.

    Args:
        predictions: Tensor containing the outputs of the model. Dimensions are
          'batch' x 'num_classes'.
        actuals: Tensor containing the ground truth labels. Dimensions are 'batch' x 'num_classes'.

    Returns:
        float: The average precision at equal recall rate across the entire batch.
    """
    aggregated_precision = 0.0
    num_videos = actuals.size(0)
    for row in range(num_videos):
        num_labels = int(actuals[row].sum())
        top_indices = predictions[row].argsort(descending=True)[:num_labels]
        item_precision = 0.0
        for label_index in top_indices:
            if predictions[row][label_index] > 0:
                item_precision += actuals[row][label_index]
        item_precision /= top_indices.size(0)
        aggregated_precision += item_precision
    aggregated_precision /= num_videos
    return aggregated_precision


import torch

def calculate_gap_torch(predictions, actuals, top_k=20, device=None):
    # 디바이스 설정이 제공되지 않았다면, predictions의 디바이스를 사용
    if device is None:
        device = predictions.device

    num_videos = predictions.size(0)
    gap = 0.0
    total_hits = 0

    for i in range(num_videos):
        video_preds = predictions[i].to(device)  # 디바이스로 이동
        video_actuals = actuals[i].to(device)  # 디바이스로 이동

        # top_k_predictions과 이에 해당하는 실제 라벨을 얻는다.
        top_k_vals, top_k_indices = torch.topk(video_preds, top_k)
        top_k_actuals = torch.gather(video_actuals, 0, top_k_indices)

        # 정답 개수를 센다.
        num_positives = top_k_actuals.sum().item()

        # num_positives가 0이면 이 비디오에 대한 계산을 건너뛴다.
        if num_positives == 0:
            continue

        # 정확도를 계산한다.
        hits = top_k_actuals.cumsum(0)
        precision_at_k = hits / torch.arange(1, top_k + 1, dtype=torch.float, device=device)

        gap += (precision_at_k * top_k_actuals).sum().item() / min(num_positives, top_k)
        total_hits += num_positives

    if num_videos > 0:
        gap /= num_videos
    else:
        gap = 0.0  # 모든 비디오가 처리되지 않은 경우

    return gap


def average_precision(output, target):
    # output과 target은 각각 모델의 예측과 실제 라벨을 나타내며, 같은 크기의 1D 텐서여야 합니다.
    # output은 확률 또는 로짓일 수 있습니다. 여기서는 확률로 가정합니다.
    # 먼저, 예측을 내림차순으로 정렬합니다.
    sorted_indices = torch.argsort(output, descending=True)
    true_positive = target[sorted_indices]
    false_positive = 1 - true_positive

    # 누적합을 계산합니다.
    tp_cumsum = true_positive.cumsum(0)
    fp_cumsum = false_positive.cumsum(0)

    # 각 임계값마다 정밀도를 계산합니다.
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

    # 각 임계값마다 재현율을 계산합니다.
    recall = tp_cumsum / (target.sum() + 1e-10)

    # AP를 계산합니다.
    ap = (precision[1:] * torch.diff(recall)).sum()
    
    return ap.item()