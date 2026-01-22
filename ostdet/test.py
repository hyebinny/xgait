import os
import argparse
import sys

import torch
from torch.utils.data import DataLoader

from util.util import load_yaml, setup_logger
from util.dataset import build_dataset
from util.model import TimmOstClassifier
from util.metric import confusion_matrix, classification_metrics_from_cm


@torch.no_grad()
def evaluate(model, loader, device, num_class):
    model.eval()
    all_pred = []
    all_y = []

    for x, y, _path in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _emb, logits = model(x)
        pred = torch.argmax(logits, dim=1)
        all_pred.append(pred)
        all_y.append(y)

    pred = torch.cat(all_pred, dim=0)
    y = torch.cat(all_y, dim=0)
    cm = confusion_matrix(pred, y, num_class=num_class)
    return classification_metrics_from_cm(cm)


def main(cfg_path: str, ckpt_path: str):
    cfg = load_yaml(cfg_path)
    out_dir = cfg["train"]["out_dir"]
    logger = setup_logger(out_dir, name="test", filename="test_log.txt")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    num_class = int(cfg["task"]["num_class"])
    test_ds, class_names = build_dataset(cfg, split="test")

    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]), pin_memory=True
    )

    mcfg = cfg["model"]
    model = TimmOstClassifier(
        backbone=mcfg["backbone"],
        pretrained=False,
        embed_dim=int(mcfg["embed_dim"]),
        num_class=num_class,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    m = evaluate(model, test_loader, device, num_class)
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"class_names={class_names}")
    logger.info("======= Test Metrics =======")
    logger.info(f"accuracy:  {m['acc']:.4f}")
    logger.info(f"precision: {m['precision']:.4f}")
    logger.info(f"recall:    {m['recall']:.4f}")
    logger.info(f"f1-socre:  {m['f1']:.4f}")    
    logger.info("============================")


def parse_args():
    parser = argparse.ArgumentParser(description="Test OST classifier")
    parser.add_argument(
        "--cfg_pth",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--ckpt_pth",
        type=str,
        default=None,
        help="Path to checkpoint .pth (default: <out_dir>/best.pth from config)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg_path = args.cfg_pth
    if args.ckpt_pth is None:
        cfg = load_yaml(cfg_path)
        ckpt_path = os.path.join(cfg["train"]["out_dir"], "best.pth")
    else:
        ckpt_path = args.ckpt_pth

    main(cfg_path, ckpt_path)