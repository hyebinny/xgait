# align_metric/extract_align_metrics.py
import os
import argparse
import json
from datetime import datetime

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from boneseg.util.util import load_yaml, read_split, ensure_dir
from boneseg.util.dataset import GNUBoneSegDataset
from boneseg.util.model import AttentionUNet

from align_metric.util.metric import (
    extract_bone_features_from_masks,
    compute_alignment_metrics_from_feats,
    add_severity_labels,
)
from align_metric.util.visualize import (
    remap_mask_to_orig,
    overlay_mask_multiclass,
)


def _to_int(x):
    if isinstance(x, torch.Tensor):
        return int(x.item())
    return int(x)


def _safe_imread(path, gray=False):
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    return cv2.imread(path, flag)


def _is_number(x) -> bool:
    # bool도 int 취급이라 제외
    if isinstance(x, (bool, np.bool_)):
        return False
    return isinstance(x, (int, float, np.integer, np.floating))


def _build_dataset(ds_name, cfg, classes):
    root = cfg["data"][ds_name]["root"]
    split = read_split(cfg["data"][ds_name]["split_json"])
    test_ids = split[cfg["data"][ds_name]["test_key"]]

    dataset = GNUBoneSegDataset(
        root=root,
        subject_ids=test_ids,
        classes=classes,
        cfg_data=cfg["data"],
        is_train=False,
    )
    return dataset, root


@torch.no_grad()
def main(cfg_pth: str, visualize: bool):
    cfg = load_yaml(cfg_pth)

    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if (device_str.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    classes = cfg["task"]["classes"]
    thr = float(cfg["task"]["threshold"])

    net_size = int(cfg["data"]["input"]["size"])
    net_hw = (net_size, net_size)  # (H, W)

    out_root = cfg["test"]["out_pth"]
    ensure_dir(out_root)

    viz_dir = os.path.join(out_root, "viz")
    viz_cfg = cfg.get("visualize", {}) or {}
    line_thick = int(viz_cfg.get("thickness", 6))
    pt_radius  = int(viz_cfg.get("radius", 6))
    
    pred_dir = os.path.join(out_root, "pred")
    if visualize:
        ensure_dir(viz_dir)
    if bool(cfg["test"].get("save_pred", False)):
        ensure_dir(pred_dir)

    log_path = os.path.join(out_root, "test_log.txt")
    f_log = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg)
        f_log.write(str(msg) + "\n")
        f_log.flush()

    # model
    ckpt_path = cfg["test"]["ckpt"]
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    model = AttentionUNet(in_ch=1, out_ch=len(classes)).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()

    # header log
    log("======== Alignment Metric Test ========")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Config: {cfg_pth}")
    log(f"Device: {device}")
    log(f"CKPT: {ckpt_path}")
    log(f"Classes: {classes}")
    log(f"Threshold: {thr}")
    log(f"Net input size: {net_size}x{net_size}")
    log(f"Visualize: {visualize}")
    log(f"Save pred: {bool(cfg['test'].get('save_pred', False))}")
    log(f"Out root: {out_root}")
    log("======================================")

    dataset_list = cfg["data"].get("dataset", [])
    assert len(dataset_list) > 0, "cfg.data.dataset is empty"

    all_results = {}
    global_rows = []

    for ds_name in dataset_list:
        assert ds_name in cfg["data"], f"Dataset '{ds_name}' not found in cfg.data"
        dataset, root = _build_dataset(ds_name, cfg, classes)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        log(f"\n[DATASET] {ds_name}")
        log(f"- root: {root}")
        log(f"- N(test): {len(dataset)}")

        ds_results = {}
        ds_rows = []

        for batch in loader:
            sid = batch["id"][0]
            img_path = batch["img_path"][0]

            orig_hw = batch.get("orig_hw", None)
            if orig_hw is not None and isinstance(orig_hw, (list, tuple)) and len(orig_hw) == 2:
                H0, W0 = _to_int(orig_hw[0]), _to_int(orig_hw[1])
            else:
                bgr_tmp = _safe_imread(img_path, gray=False)
                if bgr_tmp is None:
                    continue
                H0, W0 = bgr_tmp.shape[:2]

            x = batch["image"].to(device)  # (1,1,512,512) expected

            logits = model(x)[0]  # (C,H,W)
            prob = torch.sigmoid(logits)
            pred_net = (prob >= thr).cpu().numpy().astype(np.uint8)  # (C,512,512)

            # remap to original space
            pred_orig = np.zeros((len(classes), H0, W0), dtype=np.uint8)
            for i in range(len(classes)):
                pred_orig[i] = remap_mask_to_orig(pred_net[i], (H0, W0), net_hw)

            # metrics: ipynb 방식
            mask_dict = {classes[i]: pred_orig[i] for i in range(len(classes))}
            feats = extract_bone_features_from_masks(mask_dict)
            metrics = compute_alignment_metrics_from_feats(feats)
            metrics.update(add_severity_labels(metrics))

            ds_results[sid] = {
                "img_path": img_path,
                "orig_hw": [int(H0), int(W0)],
                "metrics": metrics,
            }
            ds_rows.append(metrics)
            global_rows.append(metrics)

            # log line
            key_show = ["PelvicTilt_deg", "HKA_L_deg", "HKA_R_deg", "mLDFA_L_deg", "mLDFA_R_deg", "MPTA_L_deg", "MPTA_R_deg"]
            msg_parts = []
            for k in key_show:
                v = metrics.get(k, np.nan)
                msg_parts.append(f"{k}={v:.2f}" if not np.isnan(v) else f"{k}=nan")
            log(f"{ds_name}/{sid}: " + " | ".join(msg_parts))

            # save pred masks
            if bool(cfg["test"].get("save_pred", False)):
                np.savez_compressed(
                    os.path.join(pred_dir, f"{ds_name}_{sid}.npz"),
                    classes=np.array(classes),
                    mask=pred_orig.astype(np.uint8),
                )

            # visualize: subject당 1장
            if visualize:
                bgr = _safe_imread(img_path, gray=False)
                if bgr is None:
                    continue
                alpha = float(cfg["test"].get("overlay_alpha", 0.35))
                bgr = overlay_mask_multiclass(bgr, pred_orig, alpha=alpha)

                # femur/tibia axis (top_mid -> bot_mid)
                for side in ["L", "R"]:
                    for bone in [f"Femur_{side}", f"Tibia_{side}"]:
                        if bone in feats and feats[bone].get("top_mid") is not None and feats[bone].get("bot_mid") is not None:
                            p1 = feats[bone]["top_mid"].astype(int)
                            p2 = feats[bone]["bot_mid"].astype(int)
                            cv2.line(bgr, tuple(p1), tuple(p2), (255, 0, 0), line_thick)
                            cv2.circle(bgr, tuple(p1), pt_radius, (0, 0, 255), -1)
                            cv2.circle(bgr, tuple(p2), pt_radius, (0, 0, 255), -1)

                # pelvis top line
                if "Pelvis" in feats:
                    p1 = feats["Pelvis"].get("pelvis_top_p1", None)
                    p2 = feats["Pelvis"].get("pelvis_top_p2", None)
                    if p1 is not None and p2 is not None:
                        p1i = tuple(p1.astype(int))
                        p2i = tuple(p2.astype(int))
                        cv2.line(bgr, p1i, p2i, (255, 0, 0), line_thick)
                        cv2.circle(bgr, p1i, pt_radius, (0, 0, 255), -1)
                        cv2.circle(bgr, p2i, pt_radius, (0, 0, 255), -1)

                cv2.imwrite(os.path.join(viz_dir, f"{ds_name}_{sid}.png"), bgr)

        all_results[ds_name] = ds_results

    # save json
    out_json = os.path.join(out_root, "align_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    log(f"\n[DONE] Saved metrics json: {out_json}")
    log(f"[DONE] Saved log: {log_path}")
    if visualize:
        log(f"[DONE] Saved viz dir: {viz_dir}")
    if bool(cfg["test"].get("save_pred", False)):
        log(f"[DONE] Saved pred dir: {pred_dir}")

    f_log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_pth", type=str, required=True, help="path to align_metric_config.yaml")
    ap.add_argument("--visualize", action="store_true", help="save overlay+axis visualizations (one image per subject)")
    args = ap.parse_args()
    main(args.cfg_pth, args.visualize)
