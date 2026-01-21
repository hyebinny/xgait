import os
import json
from typing import List, Tuple, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_gnu_align_samples(
    root: str,
    align_json: str,
    split_json: str,
    split: str,  # "train" or "test"
) -> Tuple[List[Tuple[str, int]], List[str]]:
    align = read_json(align_json)
    split_j = read_json(split_json)

    if split == "train":
        valid_subjects = set(split_j["TRAIN_SET"])
    elif split == "test":
        valid_subjects = set(split_j["TEST_SET"])
    else:
        raise ValueError(f"Unknown split: {split}")

    class_names = ["normal", "anormal"]
    label_map = {"normal": 0, "anormal": 1}

    samples: List[Tuple[str, int]] = []
    missing = 0

    for sid in align.get("NORMAL", []):
        sid = str(sid)
        if sid not in valid_subjects:
            continue
        p = os.path.join(root, sid, "xray", f"{sid}.jpg")
        if os.path.isfile(p):
            samples.append((p, label_map["normal"]))
        else:
            missing += 1

    for sid in align.get("ANORMAL", []):
        sid = str(sid)
        if sid not in valid_subjects:
            continue
        p = os.path.join(root, sid, "xray", f"{sid}.jpg")
        if os.path.isfile(p):
            samples.append((p, label_map["anormal"]))
        else:
            missing += 1

    print(f"[GNU-{split}] samples={len(samples)}, missing_xray={missing}")
    return samples, class_names


class GNUAlignDataset(Dataset):
    def __init__(self, samples, img_size, mean, std):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # PKBatchSampler용: class_to_indices
        self.class_to_indices: Dict[int, List[int]] = {}
        for i, (_, y) in enumerate(self.samples):
            self.class_to_indices.setdefault(int(y), []).append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(y, dtype=torch.long), path


def build_dataset(cfg: dict, split: str):
    dcfg = cfg["data"][split]["gnu"]
    samples, class_names = build_gnu_align_samples(
        root=dcfg["root"],
        align_json=dcfg["align_json"],
        split_json=dcfg["split_json"],
        split=split,
    )

    ds = GNUAlignDataset(
        samples=samples,
        img_size=int(cfg["data"]["img_size"]),
        mean=cfg["data"]["normalize"]["mean"],
        std=cfg["data"]["normalize"]["std"],
    )
    return ds, class_names


class PKBatchSampler(torch.utils.data.Sampler):
    """
    P*K 샘플링: 한 배치에 n_class개의 클래스, 클래스당 n_samples개 샘플.
    한 epoch 당 배치 수를 유한하게 계산해 StopIteration이 발생하도록 설계.
    """
    def __init__(self, class_to_indices, n_class, n_samples, drop_last=True):
        self.class_to_indices = {int(k): list(v) for k, v in class_to_indices.items()}
        self.n_class = int(n_class)
        self.n_samples = int(n_samples)
        self.drop_last = bool(drop_last)

        self.classes = sorted(self.class_to_indices.keys())
        if len(self.classes) == 0:
            raise ValueError("No classes in class_to_indices.")

        self.groups_per_class = {
            c: (len(self.class_to_indices[c]) // self.n_samples) if self.drop_last else max(1, (len(self.class_to_indices[c]) + self.n_samples - 1) // self.n_samples)
            for c in self.classes
        }

        min_groups = min(self.groups_per_class.values())

        self.batches_per_epoch = max(1, min_groups)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        import random

        for _ in range(self.batches_per_epoch):
            if len(self.classes) <= self.n_class:
                chosen = self.classes[:]
            else:
                chosen = random.sample(self.classes, k=self.n_class)

            batch = []
            for c in chosen:
                idxs = self.class_to_indices[c]
                if len(idxs) >= self.n_samples:
                    picked = random.sample(idxs, k=self.n_samples)
                else:
                    picked = random.choices(idxs, k=self.n_samples)
                batch.extend(picked)

            yield batch

