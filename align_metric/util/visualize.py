# align_metric/util/visualize.py
import numpy as np
import cv2


def remap_mask_to_orig(mask_net: np.ndarray, orig_hw, net_hw):
    """
    mask_net: (Hn, Wn) uint8 {0,1}
    orig_hw: (H0, W0)
    net_hw: (Hn, Wn)
    return: (H0, W0) uint8 {0,1}
    """
    H0, W0 = orig_hw
    Hn, Wn = net_hw
    if mask_net.shape[0] != Hn or mask_net.shape[1] != Wn:
        mask_net = cv2.resize(mask_net, (Wn, Hn), interpolation=cv2.INTER_NEAREST)
    mask0 = cv2.resize(mask_net, (W0, H0), interpolation=cv2.INTER_NEAREST)
    return (mask0 > 0).astype(np.uint8)


def overlay_mask_multiclass(bgr: np.ndarray, masks: np.ndarray, alpha=0.35):
    """
    bgr: (H,W,3)
    masks: (C,H,W) uint8 {0,1}
    """
    out = bgr.copy()
    H, W = bgr.shape[:2]

    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 128, 255), (255, 128, 128), (128, 255, 128),
    ]

    for c in range(masks.shape[0]):
        m = masks[c].astype(bool)
        if m.shape[0] != H or m.shape[1] != W:
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not m.any():
            continue
        color = palette[c % len(palette)]
        color_img = np.zeros_like(out, dtype=np.uint8)
        color_img[m] = color
        out = cv2.addWeighted(out, 1.0, color_img, alpha, 0)

    return out
