import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, ConcatDataset, Sampler
from PIL import Image
from torchvision import transforms

from util.util import load_json_list, load_gnu_ost_json


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(p: str) -> bool:
    return Path(p).suffix.lower() in IMG_EXTS


def build_transform(img_size: int, mean: List[float], std: List[float], train: bool):
    tf = []
    tf.append(transforms.Resize((img_size, img_size)))
    if train:
        tf.append(transforms.RandomHorizontalFlip(p=0.5))
        tf.append(transforms.RandomRotation(degrees=5))
    tf.append(transforms.ToTensor())
    tf.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(tf)


class OAI_Dataset(Dataset):
    """
    OAI root:
      root/
        implant/
        negative/
        positive/
    For 2-class:
      - negative/ -> negative
      - positive/ -> positive
      - implant/  -> use implant_negative_json: NEGATIVE list => negative else positive
    For 3-class:
      - negative/ -> negative
      - positive/ -> positive
      - implant/  -> implant
    """
    def __init__(self, root: str, num_class: int, implant_negative_json: str,
                 transform=None, class_names=None):
        self.root = root
        self.num_class = num_class
        self.transform = transform
        self.class_names = class_names

        self.implant_neg_set = load_json_list(implant_negative_json, key="NEGATIVE")

        self.samples: List[Tuple[str, int]] = []
        self._build()

        self.labels = [y for _, y in self.samples]
        self.class_to_indices = self._make_class_to_indices(self.labels, num_class)

    def _make_class_to_indices(self, labels, C):
        d = {c: [] for c in range(C)}
        for i, y in enumerate(labels):
            d[int(y)].append(i)
        return d

    def _gather_images(self, folder: str):
        if not os.path.isdir(folder):
            return []
        out = []
        for p in sorted(Path(folder).rglob("*")):
            if p.is_file() and _is_image(str(p)):
                out.append(str(p))
        return out

    def _stem_id(self, path: str):
        # e.g., ".../implant/9160026L.png" -> "9160026L"
        return Path(path).stem

    def _build(self):
        neg_dir = os.path.join(self.root, "negative")
        pos_dir = os.path.join(self.root, "positive")
        imp_dir = os.path.join(self.root, "implant")

        neg_imgs = self._gather_images(neg_dir)
        pos_imgs = self._gather_images(pos_dir)
        imp_imgs = self._gather_images(imp_dir)

        # class mapping based on class_names order:
        # 2-class: ["negative","positive"]
        # 3-class: ["negative","positive","implant"]
        name_to_id = {n: i for i, n in enumerate(self.class_names)}

        for p in neg_imgs:
            self.samples.append((p, name_to_id["negative"]))
        for p in pos_imgs:
            self.samples.append((p, name_to_id["positive"]))

        if self.num_class == 2:
            for p in imp_imgs:
                sid = self._stem_id(p)
                if sid in self.implant_neg_set:
                    self.samples.append((p, name_to_id["negative"]))
                else:
                    self.samples.append((p, name_to_id["positive"]))
        else:
            for p in imp_imgs:
                self.samples.append((p, name_to_id["implant"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long), path


class GNU_Dataset(Dataset):
    """
    GNU root:
      GNU/001/knee/001_L.png, 001_R.png ...
      GNU/001/xray/001.jpg ...
    Labels are defined in GNU_ost.json:
      POSITIVE: ["021_L", ...]
      NEGATIVE: ["001_L","001_R",...]
      IMPLANT:  ["012_L","012_R",...]
    For 2-class:
      POSITIVE -> positive
      NEGATIVE -> negative
      IMPLANT  -> use implant_negative_json: NEGATIVE list => negative else positive
    For 3-class:
      IMPLANT -> implant
    """
    def __init__(self, root, num_class, ost_json, implant_negative_json,
                 use_knee, split, split_json, transform=None, class_names=None,):
        self.root = root
        self.num_class = num_class
        self.use_knee = use_knee
        self.transform = transform
        self.class_names = class_names

        self.split = split
        self.split_json = split_json
        self.allowed_subjects = self._load_allowed_subjects()

        ost = load_gnu_ost_json(ost_json)
        self.pos_set = set(ost["POSITIVE"])
        self.neg_set = set(ost["NEGATIVE"])
        self.imp_set = set(ost["IMPLANT"])

        # implant_negative_json for GNU도 동일하게 { "NEGATIVE": ["012_L", ...] } 형태라고 가정
        self.implant_neg_set = load_json_list(implant_negative_json, key="NEGATIVE")

        self.samples: List[Tuple[str, int]] = []
        self._build()

        self.labels = [y for _, y in self.samples]
        self.class_to_indices = self._make_class_to_indices(self.labels, num_class)

    def _load_allowed_subjects(self):
        with open(self.split_json, "r") as f:
            sp = json.load(f)
        key = "TRAIN_SET" if self.split == "train" else "TEST_SET"
        return set(sp[key])

    def _make_class_to_indices(self, labels, C):
        d = {c: [] for c in range(C)}
        for i, y in enumerate(labels):
            d[int(y)].append(i)
        return d

    def _collect_knee_images(self):
        out = []
        for subj in sorted(Path(self.root).iterdir()):
            if not subj.is_dir():
                continue

            if subj.name not in self.allowed_subjects:
                continue

            knee_dir = subj / "knee"
            if knee_dir.is_dir():
                for p in sorted(knee_dir.glob("*")):
                    if p.is_file() and _is_image(str(p)):
                        out.append(str(p))
        return out

    def _collect_xray_images(self):
        out = []
        for subj in sorted(Path(self.root).iterdir()):
            if not subj.is_dir():
                continue

            if subj.name not in self.allowed_subjects:
                continue

            xray_dir = subj / "xray"
            if xray_dir.is_dir():
                for p in sorted(xray_dir.glob("*")):
                    if p.is_file() and _is_image(str(p)):
                        out.append(str(p))
        return out

    def _knee_id(self, path: str):
        # ".../knee/001_L.png" -> "001_L"
        return Path(path).stem

    def _build(self):
        name_to_id = {n: i for i, n in enumerate(self.class_names)}
        imgs = self._collect_knee_images() if self.use_knee else self._collect_xray_images()

        for p in imgs:
            if self.use_knee:
                kid = self._knee_id(p)
                if kid in self.neg_set:
                    y = name_to_id["negative"]
                elif kid in self.pos_set:
                    y = name_to_id["positive"]
                elif kid in self.imp_set:
                    if self.num_class == 3:
                        y = name_to_id["implant"]
                    else:
                        # 2-class: implant split by implant_negative_json
                        if kid in self.implant_neg_set:
                            y = name_to_id["negative"]
                        else:
                            y = name_to_id["positive"]
                else:
                    # 라벨 없는 파일은 제외 (안전)
                    continue
                self.samples.append((p, y))
            else:
                # xray는 ost_json이 knee 단위라 라벨 매칭이 불가할 가능성이 높아서 기본 제외
                # 필요하면 여기 확장
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long), path


def build_dataset(cfg: dict, split: str):
    """
    split: "train" or "test"
    """
    num_class = cfg["task"]["num_class"]
    class_names = cfg["task"]["class_names_2"] if num_class == 2 else cfg["task"]["class_names_3"]

    img_size = cfg["data"]["img_size"]
    mean = cfg["data"]["normalize"]["mean"]
    std = cfg["data"]["normalize"]["std"]

    is_train = (split == "train")
    tf = build_transform(img_size, mean, std, train=is_train)

    datasets = []
    which = cfg["data"][split]["datasets"]

    for ds_name in which:
        if ds_name.lower() == "oai":
            oai_cfg = cfg["data"][split]["oai"]
            datasets.append(
                OAI_Dataset(
                    root=oai_cfg["root"],
                    num_class=num_class,
                    implant_negative_json=oai_cfg.get("implant_negative_json", None),
                    transform=tf,
                    class_names=class_names,
                )
            )
        elif ds_name.lower() == "gnu":
            gnu_cfg = cfg["data"][split]["gnu"]
            datasets.append(
                GNU_Dataset(
                    root=gnu_cfg["root"],
                    num_class=num_class,
                    ost_json=gnu_cfg.get("ost_json", None),
                    implant_negative_json=gnu_cfg.get("implant_negative_json", None),
                    use_knee=bool(gnu_cfg.get("use_knee", True)),
                    transform=tf,
                    class_names=class_names,
                    split=split,
                    split_json=gnu_cfg.get("split_json", "dataset/GNU_json/GNU_split.json"),
                )
            )
        else:
            raise ValueError(f"Unknown dataset: {ds_name}")

    if len(datasets) == 1:
        ds = datasets[0]
    else:
        ds = ConcatDataset(datasets)

        # ConcatDataset은 내부 labels 접근이 어려우니, batch sampler를 쓰려면 wrapper가 필요함.
        # 아래 ConcatWithLabels로 감싼다.
        ds = ConcatWithLabels(datasets)

    return ds, class_names


class ConcatWithLabels(Dataset):
    """
    ConcatDataset + labels/class_to_indices를 편하게 쓰기 위한 wrapper
    """
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.cum = []
        s = 0
        for d in datasets:
            s += len(d)
            self.cum.append(s)

        # build global labels
        self.labels = []
        for d in datasets:
            self.labels.extend(getattr(d, "labels"))
        self.num_class = len(set(self.labels))
        self.class_to_indices = self._make_class_to_indices(self.labels, self.num_class)

    def _make_class_to_indices(self, labels, C):
        d = {c: [] for c in range(C)}
        for i, y in enumerate(labels):
            d[int(y)].append(i)
        return d

    def __len__(self):
        return self.cum[-1]

    def _locate(self, idx: int):
        for di, c in enumerate(self.cum):
            if idx < c:
                prev = 0 if di == 0 else self.cum[di - 1]
                return di, idx - prev
        raise IndexError

    def __getitem__(self, idx):
        di, li = self._locate(idx)
        return self.datasets[di][li]


class PKBatchSampler(Sampler[List[int]]):
    """
    n_class classes per batch, n_samples per class
    """
    def __init__(self, class_to_indices: Dict[int, List[int]], n_class: int, n_samples: int, drop_last: bool = True):
        self.class_to_indices = {k: v[:] for k, v in class_to_indices.items() if len(v) > 0}
        self.classes = sorted(list(self.class_to_indices.keys()))
        self.n_class = n_class
        self.n_samples = n_samples
        self.drop_last = drop_last

        self.num_batches = self._estimate_num_batches()

    def _estimate_num_batches(self):
        # rough estimate: total available / batch_size
        total = sum(len(v) for v in self.class_to_indices.values())
        bs = self.n_class * self.n_samples
        return max(1, total // bs)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        import random

        # shuffle indices per class
        pools = {c: v[:] for c, v in self.class_to_indices.items()}
        for c in pools:
            random.shuffle(pools[c])

        bs = self.n_class * self.n_samples

        for _ in range(self.num_batches):
            # sample classes
            chosen = random.sample(self.classes, k=min(self.n_class, len(self.classes)))
            batch = []
            for c in chosen:
                idxs = pools[c]
                if len(idxs) < self.n_samples:
                    # refill with reshuffle
                    refill = self.class_to_indices[c][:]
                    random.shuffle(refill)
                    idxs.extend(refill)
                batch.extend(idxs[: self.n_samples])
                pools[c] = idxs[self.n_samples :]

            if len(batch) < bs:
                if self.drop_last:
                    continue
            yield batch
