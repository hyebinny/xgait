import os
import json
import cv2
import numpy as np

def visualize_labelme_polygons(
    img_path: str,
    json_path: str,
    out_path: str,
    alpha: float = 0.35,
    line_thickness: int = 2,
    font_scale: float = 0.7,
    font_thickness: int = 2,
):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    h, w = img.shape[:2]

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    if not shapes:
        raise ValueError(f"No shapes found in json: {json_path}")

    overlay = img.copy()

    def color_for_label(label: str):
        v = abs(hash(label)) % (256**3)
        b = (v // (256**0)) % 256
        g = (v // (256**1)) % 256
        r = (v // (256**2)) % 256
        return (int(b), int(g), int(r))

    for sh in shapes:
        label = sh.get("label", "unknown")
        pts = sh.get("points", [])
        shape_type = sh.get("shape_type", "polygon")

        if not pts or shape_type != "polygon":
            continue

        poly = np.array([[float(x), float(y)] for x, y in pts], dtype=np.float32)
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        poly_i = poly.astype(np.int32).reshape((-1, 1, 2))

        color = color_for_label(label)

        cv2.fillPoly(overlay, [poly_i], color)
        cv2.polylines(overlay, [poly_i], True, color, thickness=line_thickness)

        cx = int(np.mean(poly[:, 0]))
        cy = int(np.mean(poly[:, 1]))
        cv2.putText(
            overlay,
            label,
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    vis = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not cv2.imwrite(out_path, vis):
        raise RuntimeError(f"Failed to write output: {out_path}")

# =========================
# batch processing
# =========================
if __name__ == "__main__":
    ROOT = "/mnt/d/xgait/dataset/GNU"
    OUT_DIR = "/mnt/d/xgait/boneseg/bone_gt_viz"

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
