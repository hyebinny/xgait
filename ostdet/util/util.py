import os
import time
import json
import random
import logging
from pathlib import Path

import numpy as np
import torch
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def setup_logger(out_dir: str, name: str = "train"):
    ensure_dir(out_dir)
    log_path = os.path.join(out_dir, "log.txt")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class Timer:
    def __init__(self):
        self.t0 = None

    def start(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        if self.t0 is None:
            return 0.0
        return time.time() - self.t0


def save_checkpoint(path: str, payload: dict):
    ensure_dir(os.path.dirname(path))
    torch.save(payload, path)


def load_json_list(path: str, key: str = "NEGATIVE"):
    """
    expected:
    { "NEGATIVE": ["9160026L", ...] }
    """
    if path is None:
        return set()
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        obj = json.load(f)
    arr = obj.get(key, [])
    return set(arr)


def load_gnu_ost_json(path: str):
    """
    expected:
    { "POSITIVE": [...], "NEGATIVE": [...], "IMPLANT": [...] }
    values like "001_L"
    """
    if path is None or (not os.path.exists(path)):
        return {"POSITIVE": [], "NEGATIVE": [], "IMPLANT": []}
    with open(path, "r") as f:
        obj = json.load(f)
    return {
        "POSITIVE": obj.get("POSITIVE", []),
        "NEGATIVE": obj.get("NEGATIVE", []),
        "IMPLANT": obj.get("IMPLANT", []),
    }
