import os
import argparse
import json
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from boneseg.util.util import load_yaml, read_split, ensure_dir
from boneseg.util.dataset import GNUBoneSegDataset
from boneseg.util.model import AttentionUNet


# =========================
# Geometry utilities
# =========================

def compute_principal_axis(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    center = pts.mean(axis=0)
    pts_c = pts - center

    _, _, vt = np.linalg.svd(pts_c, full_matrices=False)
    direction = vt[0]
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    return center, direction


def angle_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


# =========================
# Alignment metrics
# =========================

def compute_HKA(femur_axis, tibia_axis):
    return angle_between(femur_axis[1], tibia_axis[1])


def compute_pelvic_tilt(pelvis_axis):
    return angle_between(pelvis_axis[1], np.array([1.0, 0.0]))


def compute_mLDFA(femur_axis):
    return angle_between(femur_axis[1], np.array([0.0, 1.0]))


def compute_MPTA(tibia_axis):
    return angle_between(tibia_axis[1], np.array([0.0, 1.0]))


# =========================
# Visualization
# =========================

def draw_axis(img, center, direction, color, length=300):
    c = tuple(center.astype(int))
    d = direction * length
    p1 = (int(c[0] - d[0]), int(c[1] - d[1]))
    p2 = (int(c[0] + d[0]), int(c[1] + d[1]))
    cv2.line(img, p1, p2, color, 2)


# =========================
# Main
# =========================

@torch.no_grad()
def main(cfg_pth, visualize):
    cfg = load_yaml(cfg_pth)
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    classes = cfg["task"]["classes"]
    thr = float(cfg["task"]["threshold"])

    out_root = cfg["test"]["out_pth"]
    ensure_dir(out_root)
    if visualize:
        ensure_dir(os.path.join(out_root, "viz"))

    ckpt_path = cfg["test"]["ckpt"]
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    model = AttentionUNet(in_ch=1, out_ch=len(classes)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()

    all_results = {}

    for ds_name in ["gnu", "snu"]:
        if ds_name not in cfg["data"]:
            continue

        root = cfg["data"][ds_name]["root"]
        split = read_split(cfg["data"][ds_name]["split_json"])
        test_ids = split[cfg["data"][ds_name]["test_key"]]

        dataset = GNUBoneSegDataset(
            root=root,
            ids=test_ids,
            classes=classes,
            data_cfg=cfg["data"],
            is_train=False,
        )

        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        for batch in loader:
            sid = batch["id"][0]
            img_path = batch["img_path"][0]
            x = batch["image"].to(device)

            pred = (torch.sigmoid(model(x))[0] >= thr).cpu().numpy()

            axes = {}
            for i, name in enumerate(classes):
                axis = compute_principal_axis(pred[i])
                if axis is not None:
                    axes[name] = axis

            metrics = {}
            if "Femur_L" in axes and "Tibia_L" in axes:
                metrics["HKA_L"] = compute_HKA(axes["Femur_L"], axes["Tibia_L"])
                metrics["mLDFA_L"] = compute_mLDFA(axes["Femur_L"])
                metrics["MPTA_L"] = compute_MPTA(axes["Tibia_L"])

            if "Pelvis" in axes:
                metrics["PelvicTilt"] = compute_pelvic_tilt(axes["Pelvis"])

            all_results[sid] = metrics

            if visualize:
                bgr = cv2.imread(img_path)
                for _, (c, d) in axes.items():
                    draw_axis(bgr, c, d, (0, 255, 0))
                for i, (k, v) in enumerate(metrics.items()):
                    cv2.putText(
                        bgr, f"{k}: {v:.2f}",
                        (20, 40 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2
                    )
                cv2.imwrite(os.path.join(out_root, "viz", f"{sid}.png"), bgr)

    with open(os.path.join(out_root, "align_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[DONE] Alignment metrics saved to {out_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_pth", type=str, required=True)
    ap.add_argument("--visualize", action="store_true")
    args = ap.parse_args()

    main(args.cfg_pth, args.visualize)
