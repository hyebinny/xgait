import torch
import torch.nn.functional as F


def ce_loss(logits: torch.Tensor, y: torch.Tensor):
    return F.cross_entropy(logits, y)


def _pairwise_dist(x: torch.Tensor):
    """
    x: (B,D) -> dist: (B,B)
    """
    # squared euclidean
    xx = (x * x).sum(dim=1, keepdim=True)
    dist2 = xx + xx.t() - 2.0 * (x @ x.t())
    dist2 = torch.clamp(dist2, min=0.0)
    return torch.sqrt(dist2 + 1e-12)


def triplet_loss_all(emb: torch.Tensor, y: torch.Tensor, margin: float):
    """
    모든 (a,p,n) 조합을 다 쓰면 너무 무거워질 수 있어서,
    (a,p)마다 가장 가까운 n(=hard negative)을 쓰는 형태로 가볍게 구성.
    """
    dist = _pairwise_dist(emb)
    B = emb.size(0)

    loss_sum = 0.0
    cnt = 0

    for i in range(B):
        yi = y[i]
        pos_mask = (y == yi) & (torch.arange(B, device=y.device) != i)
        neg_mask = (y != yi)
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        pos_d = dist[i][pos_mask]                 # (P,)
        neg_d = dist[i][neg_mask]                 # (N,)
        # for each positive, use hardest negative (smallest distance)
        hard_neg = neg_d.min()
        # sum over positives
        l = F.relu(pos_d - hard_neg + margin).mean()
        loss_sum += l
        cnt += 1

    if cnt == 0:
        return emb.new_tensor(0.0)
    return loss_sum / cnt


def triplet_loss_hard(emb: torch.Tensor, y: torch.Tensor, margin: float):
    """
    batch-hard triplet:
    hardest positive(=largest dist) + hardest negative(=smallest dist)
    """
    dist = _pairwise_dist(emb)
    B = emb.size(0)

    loss_sum = 0.0
    cnt = 0

    for i in range(B):
        yi = y[i]
        pos_mask = (y == yi) & (torch.arange(B, device=y.device) != i)
        neg_mask = (y != yi)
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        hardest_pos = dist[i][pos_mask].max()
        hardest_neg = dist[i][neg_mask].min()

        loss_sum += F.relu(hardest_pos - hardest_neg + margin)
        cnt += 1

    if cnt == 0:
        return emb.new_tensor(0.0)
    return loss_sum / cnt
