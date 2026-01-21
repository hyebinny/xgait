import torch


def confusion_matrix(pred, y, num_class: int):
    cm = torch.zeros((num_class, num_class), dtype=torch.long, device=pred.device)
    for p, t in zip(pred.view(-1), y.view(-1)):
        cm[int(t), int(p)] += 1
    return cm


def classification_metrics_from_cm(cm):
    cm = cm.to(torch.float32)
    tp = torch.diag(cm)

    acc = (tp.sum() / cm.sum().clamp(min=1.0)).item()
    precision = (tp / cm.sum(0).clamp(min=1.0)).mean().item()
    recall = (tp / cm.sum(1).clamp(min=1.0)).mean().item()
    f1 = (2 * precision * recall / max(1e-12, precision + recall))
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}
