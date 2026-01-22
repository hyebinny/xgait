import torch

@torch.no_grad()
def iou_per_class(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7):
    # pred, gt: [B,C,H,W] in {0,1}
    B, C = pred.shape[:2]
    pred = pred.view(B, C, -1)
    gt = gt.view(B, C, -1)

    inter = (pred & gt).sum(dim=-1).float()
    union = (pred | gt).sum(dim=-1).float()
    iou = (inter + eps) / (union + eps)  # [B,C]
    return iou.mean(dim=0)               # [C]

@torch.no_grad()
def mean_iou(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7):
    return iou_per_class(pred, gt, eps=eps).mean().item()
