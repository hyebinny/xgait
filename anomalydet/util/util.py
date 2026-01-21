import os
import yaml
import time
import random
import logging

import numpy as np
import torch

from pathlib import Path

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def setup_logger(out_dir: str, name="run", filename="log.txt"):
    import logging
    import os

    ensure_dir(out_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(os.path.join(out_dir, filename))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Timer:
    def __init__(self):
        self.t0 = None

    def start(self):
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0 if self.t0 is not None else 0.0


def save_checkpoint(path: str, payload: dict):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
