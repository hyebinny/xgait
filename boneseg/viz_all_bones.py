import os
import json
import cv2
import numpy as np

from util.visualize import visualize_labelme_polygons

# =========================
# batch processing
# =========================
if __name__ == "__main__":
    ROOT = "/mnt/d/hyebin/xgait/dataset/GNU"
    OUT_DIR = "/mnt/d/hyebin/xgait/boneseg/bone_gt_viz"

    os.makedirs(OUT_DIR, exist_ok=True)

    subjects = sorted(os.listdir(ROOT))
    failed = []

    for sid in subjects:
        subject_dir = os.path.join(ROOT, sid)
        if not os.path.isdir(subject_dir):
            continue

        img_path = os.path.join(subject_dir, "xray", f"{sid}.jpg")
        json_path = os.path.join(subject_dir, "label", f"{sid}.json")
        out_path = os.path.join(OUT_DIR, f"{sid}.png")

        if not (os.path.isfile(img_path) and os.path.isfile(json_path)):
            failed.append(sid)
            continue

        try:
            visualize_labelme_polygons(img_path, json_path, out_path)
            print(f"[OK] {sid}")
        except Exception as e:
            print(f"[FAIL] {sid}: {e}")
            failed.append(sid)

    print("\n=== DONE ===")
    print(f"Total subjects: {len(subjects)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print("Failed IDs:", failed)
