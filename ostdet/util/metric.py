import torch


@torch.no_grad()
def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_class: int):
    """
    pred, target: (N,)
    """
    cm = torch.zeros((num_class, num_class), dtype=torch.long, device=pred.device)
    for t, p in zip(target, pred):
        cm[t.long(), p.long()] += 1
    return cm


@torch.no_grad()
def classification_metrics_from_cm(cm: torch.Tensor):
    """
    returns macro precision/recall/f1 + accuracy
    cm: (C,C), rows=GT, cols=Pred
    """
    C = cm.size(0)
    tp = cm.diag().float()
    support = cm.sum(dim=1).float()          # GT count per class
    pred_count = cm.sum(dim=0).float()       # predicted count per class

    precision = tp / (pred_count + 1e-12)
    recall = tp / (support + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    acc = tp.sum() / (cm.sum().float() + 1e-12)

    # macro (ignore classes with 0 support? 보통 GT 0인 클래스는 macro에서 제외)
    valid = support > 0
    if valid.any():
        precision_m = precision[valid].mean()
        recall_m = recall[valid].mean()
        f1_m = f1[valid].mean()
    else:
        precision_m = precision.mean()
        recall_m = recall.mean()
        f1_m = f1.mean()

    return {
        "acc": acc.item(),
        "precision": precision_m.item(),
        "recall": recall_m.item(),
        "f1": f1_m.item(),
    }
