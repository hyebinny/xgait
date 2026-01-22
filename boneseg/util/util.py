import os
import json
import yaml
import random
import time
import numpy as np
import cv2
import torch
from datetime import datetime
from typing import Optional, List, Tuple

# -------------------------
# config / seed / io
# -------------------------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def now_str():
    return time.strftime("%Y%m%d_%H%M%S")

def make_run_dir(cfg: dict) -> dict:
    out_root = cfg["train"]["out_dir"]
    exp_name = cfg["train"]["exp_name"]
    run_dir = os.path.join(out_root, exp_name)

    ckpt_dir = os.path.join(run_dir, "ckpt")
    log_dir = os.path.join(run_dir, "log")
    pred_dir = os.path.join(run_dir, "preds")

    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)
    ensure_dir(pred_dir)

    return {"run_dir": run_dir, "ckpt_dir": ckpt_dir, "log_dir": log_dir, "pred_dir": pred_dir}

def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def read_split(split_json: str) -> dict:
    with open(split_json, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------
# logging
# -------------------------
class SimpleLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        ensure_dir(os.path.dirname(log_path))

    def log(self, msg: str):
        s = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(s)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(s + "\n")

# -------------------------
# ckpt
# -------------------------
def save_ckpt(path: str, model, optimizer=None, scheduler=None, epoch=0, best_score=None):
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)

def load_ckpt(path: str, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt

# -------------------------
# preprocess / aug
# -------------------------
def preprocess_xray_gray(gray: np.ndarray, cfg_pre: dict) -> np.ndarray:
    # gray: uint8 [H,W]
    if not cfg_pre.get("enable", True):
        return gray

    mode = cfg_pre.get("mode", "none")
    if mode == "clahe_unsharp":
        clip = float(cfg_pre.get("clahe_clip", 2.0))
        grid = int(cfg_pre.get("clahe_grid", 8))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        g = clahe.apply(gray)

        sigma = float(cfg_pre.get("unsharp_sigma", 1.0))
        amount = float(cfg_pre.get("unsharp_amount", 1.5))
        blur = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma, sigmaY=sigma)
        sharp = cv2.addWeighted(g, 1.0 + amount, blur, -amount, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    return gray

def resize_img_and_masks(gray: np.ndarray, masks: np.ndarray, size: int):
    # gray: [H,W], masks: [C,H,W] uint8(0/1)
    g2 = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)
    m2 = []
    for c in range(masks.shape[0]):
        m = cv2.resize(masks[c], (size, size), interpolation=cv2.INTER_NEAREST)
        m2.append(m)
    m2 = np.stack(m2, axis=0)
    return g2, m2

def swap_lr_channels(masks: np.ndarray, classes: list):
    # masks: [C,H,W], classes fixed order
    idx = {name: i for i, name in enumerate(classes)}
    pairs = [("Tibia_L", "Tibia_R"), ("Fibula_L", "Fibula_R"), ("Femur_L", "Femur_R")]
    out = masks.copy()
    for a, b in pairs:
        if a in idx and b in idx:
            ia, ib = idx[a], idx[b]
            out[ia], out[ib] = masks[ib].copy(), masks[ia].copy()
    return out

def random_affine(gray: np.ndarray, masks: np.ndarray, cfg_aug: dict, classes: list):
    """
    gray: [S,S] uint8
    masks: [C,S,S] uint8 (0/1)
    return: (gray_aug, masks_aug)
    """
    # aug 비활성화면 그대로 반환
    if cfg_aug is None or (cfg_aug.get("enable", True) is False):
        return gray, masks

    S = int(gray.shape[0])

    deg = float(cfg_aug.get("rotate_deg", 0))
    trans = float(cfg_aug.get("translate", 0))
    smin = float(cfg_aug.get("scale_min", 1.0))
    smax = float(cfg_aug.get("scale_max", 1.0))

    angle = random.uniform(-deg, deg) if deg > 0 else 0.0
    scale = random.uniform(smin, smax) if (smin != 1.0 or smax != 1.0) else 1.0
    tx = random.uniform(-trans, trans) * S if trans > 0 else 0.0
    ty = random.uniform(-trans, trans) * S if trans > 0 else 0.0

    M = cv2.getRotationMatrix2D((S / 2, S / 2), angle, scale)
    M[:, 2] += (tx, ty)

    g2 = cv2.warpAffine(
        gray, M, (S, S),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    m2 = []
    for c in range(masks.shape[0]):
        mc = cv2.warpAffine(
            masks[c], M, (S, S),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        m2.append(mc)
    m2 = np.stack(m2, axis=0)

    # hflip + 좌우 채널 swap
    if cfg_aug.get("hflip", True) and random.random() < 0.5:
        g2 = np.ascontiguousarray(g2[:, ::-1])
        m2 = np.ascontiguousarray(m2[:, :, ::-1])
        m2 = swap_lr_channels(m2, classes)

    # brightness / contrast (image only)
    b = float(cfg_aug.get("brightness", 0.0))
    c = float(cfg_aug.get("contrast", 0.0))

    if b > 0:
        delta = random.uniform(-b, b) * 255.0
    else:
        delta = 0.0

    if c > 0:
        factor = 1.0 + random.uniform(-c, c)
    else:
        factor = 1.0

    g2 = np.clip(g2.astype(np.float32) * factor + delta, 0, 255).astype(np.uint8)

    return g2, m2

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class Logger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def info(self, msg: str):
        line = f"[{timestamp()}] {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def overlay_masks_on_bgr(
    bgr: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.35,
    colors: Optional[List[Tuple[int, int, int]]] = None,
):
    """
    bgr: (H, W, 3) uint8
    masks: (C, H, W) uint8/bool (0/1 or 0/255)
    alpha: overlay weight (overlay = alpha, original = 1-alpha)
    colors: list of BGR colors per class (optional)
    """
    if bgr is None:
        raise ValueError("bgr is None")
    if masks is None:
        raise ValueError("masks is None")

    if masks.dtype != np.uint8 and masks.dtype != np.bool_:
        masks = masks.astype(np.uint8)

    C, H, W = masks.shape
    if bgr.shape[0] != H or bgr.shape[1] != W:
        raise ValueError(f"Shape mismatch: bgr={bgr.shape}, masks={masks.shape}")

    overlay = bgr.copy()

    # 기본 팔레트 (BGR)
    if colors is None:
        colors = [
            (255, 0, 0),     # blue
            (0, 255, 0),     # green
            (0, 0, 255),     # red
            (255, 255, 0),   # cyan
            (255, 0, 255),   # magenta
            (0, 255, 255),   # yellow
            (128, 128, 255), # light-red
        ]

    for c in range(C):
        m = masks[c]
        if m.dtype == np.bool_:
            mask = m
        else:
            mask = m > 0

        if not np.any(mask):
            continue

        color = colors[c % len(colors)]
        overlay[mask] = color

    vis = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)
    return vis