import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .util import preprocess_xray_gray, resize_img_and_masks, random_affine

def labelme_json_to_multimask(json_path: str, classes: list, H: int, W: int):
    # output: [C,H,W] uint8 0/1
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    masks = np.zeros((len(classes), H, W), dtype=np.uint8)
    shapes = data.get("shapes", [])
    if not shapes:
        return masks

    name_to_idx = {n: i for i, n in enumerate(classes)}

    for sh in shapes:
        label = sh.get("label", "")
        stype = sh.get("shape_type", "polygon")
        pts = sh.get("points", [])

        if stype != "polygon" or not pts:
            continue
        if label not in name_to_idx:
            # 모르는 라벨은 무시
            continue

        poly = np.array([[float(x), float(y)] for x, y in pts], dtype=np.float32)
        poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
        poly_i = poly.astype(np.int32).reshape((-1, 1, 2))

        idx = name_to_idx[label]
        cv2.fillPoly(masks[idx], [poly_i], 1)

    return masks

class GNUBoneSegDataset(Dataset):
    def __init__(self, root: str, subject_ids: list, classes: list, cfg_data: dict, is_train: bool):
        self.root = root
        self.ids = subject_ids
        self.classes = classes
        self.cfg_data = cfg_data
        self.is_train = is_train

        self.size = int(cfg_data["input"]["size"])
        self.pre_cfg = cfg_data.get("preprocess", {})
        self.aug_cfg = cfg_data.get("aug", {})

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        img_path = os.path.join(self.root, sid, "xray", f"{sid}.jpg")
        json_path = os.path.join(self.root, sid, "label", f"{sid}.json")

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(img_path)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = preprocess_xray_gray(gray, self.pre_cfg)

        H, W = gray.shape[:2]
        masks = labelme_json_to_multimask(json_path, self.classes, H, W)  # [C,H,W]

        # resize to network input
        gray, masks = resize_img_and_masks(gray, masks, self.size)

        # augment (train only)
        if self.is_train and self.aug_cfg.get("enable", True):
            gray, masks = random_affine(gray, masks, self.aug_cfg, self.classes)

        # normalize
        x = gray.astype(np.float32)
        if self.cfg_data["input"].get("normalize", True):
            x = x / 255.0
        x = np.expand_dims(x, axis=0)  # [1,H,W]

        y = masks.astype(np.float32)   # [C,H,W]

        return {
            "id": sid,
            "image": torch.from_numpy(x),
            "mask": torch.from_numpy(y),
            "img_path": img_path,
            "orig_hw": (H, W),
        }

