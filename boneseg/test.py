import os
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from util.util import (
    load_yaml,
    make_run_dir,
    Logger,
    read_split,
    load_ckpt,
    ensure_dir,
    overlay_masks_on_bgr,
)
from util.dataset import GNUBoneSegDataset
from util.model import AttentionUNet
from util.metric import iou_per_class


@torch.no_grad()
def main(cfg_path: str, visualize: bool = False):
    cfg = load_yaml(cfg_path)

    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")

    run = make_run_dir(cfg)

    log_path = os.path.join(run["log_dir"], "test_log.txt")
    logger = Logger(log_path)

    classes = cfg["task"]["classes"]
    num_class = int(cfg["task"]["num_class"])
    thr = float(cfg["task"].get("threshold", 0.5))

    split = read_split(cfg["data"]["gnu"]["split_json"])
    test_ids = list(split[cfg["data"]["gnu"]["test_key"]])  # TEST_SET

    ds_te = GNUBoneSegDataset(cfg["data"]["gnu"]["root"], test_ids, classes, cfg["data"], is_train=False)
    dl_te = DataLoader(
        ds_te,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
    )

    model = AttentionUNet(in_ch=1, out_ch=num_class).to(device)

    ckpt = cfg.get("test", {}).get("ckpt", "")
    if not ckpt:
        ckpt = os.path.join(run["ckpt_dir"], "final.pth")
        if not os.path.exists(ckpt):
            ckpt = os.path.join(run["ckpt_dir"], "last.pth")
    load_ckpt(ckpt, model, map_location=device)

    logger.info("======== Test Setup ========")
    logger.info(f"Config: {cfg_path}")
    logger.info(f"Run dir: {run['run_dir']}")
    logger.info(f"Device: {device}")
    logger.info(f"Loaded ckpt: {ckpt}")
    logger.info(f"Test samples: {len(ds_te)}")
    logger.info(f"Num class: {num_class} / class_names={classes}")
    logger.info(f"Threshold: {thr}")
    logger.info(f"Visualize: {visualize}")
    logger.info("==================================")

    save_pred = bool(cfg.get("test", {}).get("save_pred", True))
    pred_dir = cfg.get("test", {}).get("pred_dir", "")
    if not pred_dir:
        pred_dir = run["pred_dir"]
    alpha = float(cfg.get("test", {}).get("overlay_alpha", 0.35))

    if visualize and save_pred:
        vis_dir = os.path.join(pred_dir, "viz")
        ensure_dir(vis_dir)

    model.eval()

    iou_sum = torch.zeros(len(classes), dtype=torch.float64)

    for batch in dl_te:
        sid = batch["id"][0]
        x = batch["image"].to(device, non_blocking=True)
        gt = batch["mask"].to(device, non_blocking=True)

        logits = model(x)
        pred = (torch.sigmoid(logits) >= thr)
        gt_bin = (gt >= 0.5)

        iou_c = iou_per_class(pred.cpu(), gt_bin.cpu())  # [C]
        iou_sum += iou_c.double()

        if visualize and save_pred:
            img_path = batch["img_path"][0]

            # orig_hw 안전 처리
            orig_hw = batch["orig_hw"]
            if isinstance(orig_hw, (list, tuple)) and len(orig_hw) == 1:
                orig_hw = orig_hw[0]
            if isinstance(orig_hw, (list, tuple)) and len(orig_hw) == 2:
                H, W = orig_hw
            else:
                bgr_tmp = cv2.imread(img_path, cv2.IMREAD_COLOR)
                H, W = bgr_tmp.shape[:2]

            if isinstance(H, torch.Tensor): H = int(H.item())
            if isinstance(W, torch.Tensor): W = int(W.item())

            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue

            pm = pred[0].cpu().numpy().astype("uint8")  # [C,512,512]
            pm_up = []
            for c in range(pm.shape[0]):
                m = cv2.resize(pm[c], (W, H), interpolation=cv2.INTER_NEAREST)
                pm_up.append(m)
            pm_up = np.stack(pm_up, axis=0)  # [C,H,W]

            vis = overlay_masks_on_bgr(bgr, pm_up, alpha=alpha)
            cv2.imwrite(os.path.join(pred_dir, "viz", f"{sid}_overlay.png"), vis)

    iou_mean = (iou_sum / max(1, len(dl_te))).numpy()
    miou = float(iou_mean.mean())

    logger.info("=== TEST IoU per class ===")
    for name, v in zip(classes, iou_mean):
        logger.info(f"{name}: {float(v):.4f}")
    logger.info(f"mIoU: {miou:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_pth", type=str, required=True)
    ap.add_argument("--visualize", action="store_true")
    args = ap.parse_args()

    main(args.cfg_pth, visualize=args.visualize)
