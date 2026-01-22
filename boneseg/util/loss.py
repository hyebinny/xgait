import torch
import torch.nn as nn

def dice_loss_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5):
    # logits, targets: [B,C,H,W], targets in {0,1}
    probs = torch.sigmoid(logits)
    B, C = probs.shape[:2]
    probs = probs.view(B, C, -1)
    targets = targets.view(B, C, -1)

    inter = (probs * targets).sum(dim=-1)
    den = probs.sum(dim=-1) + targets.sum(dim=-1)
    dice = (2 * inter + smooth) / (den + smooth)
    return 1.0 - dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_w = float(bce_weight)
        self.dice_w = float(dice_weight)
        self.smooth = float(smooth)

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        dice = dice_loss_logits(logits, targets, smooth=self.smooth)
        return self.bce_w * bce + self.dice_w * dice
