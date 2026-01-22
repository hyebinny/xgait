import os
import argparse
import time

import torch
from torch.utils.data import DataLoader

from util.util import (
    load_yaml,
    set_seed,
    make_run_dir,
    Logger,
    read_split,
    save_ckpt,
)
from util.dataset import GNUBoneSegDataset
from util.model import AttentionUNet
from util.loss import BCEDiceLoss
from util.metric import iou_per_class  # per-class IoU


def build_optimizer_and_scheduler(cfg, model):
    optim_cfg = cfg.get("optim", {})
    name = optim_cfg.get("name", "adamw").lower()
    lr = float(optim_cfg.get("lr", 1e-3))
    wd = float(optim_cfg.get("weight_decay", 0.0))

    if name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

    sch_cfg = optim_cfg.get("scheduler", {})
    sch_name = sch_cfg.get("name", "none").lower()

    scheduler = None
    if sch_name == "cosine":
        epochs = int(cfg["train"]["epochs"])
        min_lr = float(sch_cfg.get("min_lr", 1e-6))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )

    return optimizer, scheduler


@torch.no_grad()
def eval_on_loader(model, loader, device, thr):
    """
    return:
      iou_mean_per_class: np.array [C]
      miou: float
    """
    model.eval()
    C = None
    iou_sum = None
    n = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        gt = batch["mask"].to(device, non_blocking=True)

        logits = model(x)
        pred = (torch.sigmoid(logits) >= thr)
        gt_bin = (gt >= 0.5)

        iou_c = iou_per_class(pred.cpu(), gt_bin.cpu())  # [C]
        if iou_sum is None:
            C = iou_c.numel()
            iou_sum = torch.zeros(C, dtype=torch.float64)
        iou_sum += iou_c.double()
        n += 1

    iou_mean = (iou_sum / max(1, n)).numpy()
    miou = float(iou_mean.mean())
    return iou_mean, miou


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    set_seed(int(cfg.get("seed", 42)))

    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")

    run = make_run_dir(cfg)

    # log file name
    log_path = os.path.join(run["log_dir"], "log.txt")
    logger = Logger(log_path)

    classes = cfg["task"]["classes"]
    num_class = int(cfg["task"]["num_class"])
    thr = float(cfg["task"].get("threshold", 0.5))

    split = read_split(cfg["data"]["gnu"]["split_json"])
    train_ids = list(split[cfg["data"]["gnu"]["train_key"]])
    test_ids = list(split[cfg["data"]["gnu"]["test_key"]])

    # loaders
    ds_tr = GNUBoneSegDataset(cfg["data"]["gnu"]["root"], train_ids, classes, cfg["data"], is_train=True)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )

    ds_te = GNUBoneSegDataset(cfg["data"]["gnu"]["root"], test_ids, classes, cfg["data"], is_train=False)
    dl_te = DataLoader(
        ds_te,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
    )

    # model / loss / optim
    model = AttentionUNet(in_ch=1, out_ch=num_class).to(device)
    criterion = BCEDiceLoss(**cfg["loss"]).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)

    use_fp16 = bool(cfg["train"].get("fp16", False)) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # schedule params
    epochs = int(cfg["train"]["epochs"])
    log_epoch = int(cfg["train"].get("log_epoch", 5))
    eval_freq = int(cfg["train"].get("eval_freq", 0))      # 0이면 eval 안 함
    save_epoch = int(cfg["train"].get("save_epoch", 0))    # 0이면 주기 저장 안 함

    # setup logs
    logger.info("======== Experiment Setup ========")
    logger.info(f"Config: {cfg_path}")
    logger.info(f"Run dir: {run['run_dir']}")
    logger.info(f"Device: {device}")
    logger.info(f"Num class: {num_class} / class_names={classes}")
    logger.info(f"Train samples: {len(ds_tr)}")
    logger.info(f"Test samples: {len(ds_te)}")
    logger.info("Backbone: AttentionUNet")
    logger.info(f"Epochs: {epochs}, lr={cfg['optim']['lr']}, wd={cfg['optim'].get('weight_decay', 0.0)}")
    logger.info(f"Batch size: {cfg['train']['batch_size']}")
    logger.info(f"FP16: {use_fp16}")
    logger.info(f"Loss: BCE(w={cfg['loss']['bce_weight']}) + Dice(w={cfg['loss']['dice_weight']}, smooth={cfg['loss'].get('smooth', 1e-5)})")
    logger.info(f"log_epoch={log_epoch}, eval_freq={eval_freq}, save_epoch={save_epoch}")
    logger.info("==================================")

    best_miou = -1.0
    t0_all = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total_loss = 0.0

        for batch in dl_tr:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_fp16):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.perf_counter() - t0
        avg_loss = total_loss / max(1, len(dl_tr))

        # loss log
        if epoch % log_epoch == 0:
            logger.info(f"[Epoch {epoch:04d}/{epochs}] loss={avg_loss:.6f} | epoch_time={epoch_time:.1f}s")

        # always keep last
        last_path = os.path.join(run["ckpt_dir"], "last.pth")
        save_ckpt(last_path, model, optimizer, scheduler, epoch=epoch, best_score=best_miou)

        # periodic eval on TEST_SET
        if eval_freq > 0 and (epoch % eval_freq == 0):
            te0 = time.perf_counter()
            iou_mean, miou = eval_on_loader(model, dl_te, device, thr)
            eval_time = time.perf_counter() - te0

            # short one-line + per-class line
            logger.info(
                f"[Epoch {epoch:04d}/{epochs}] "
                f"test(mIoU={miou:.4f}) | epoch_time={epoch_time:.1f}s eval_time={eval_time:.1f}s"
            )
            per_cls = " | ".join([f"{n}:{v:.3f}" for n, v in zip(classes, iou_mean)])
            logger.info(f"[Epoch {epoch:04d}] test IoU per class -> {per_cls}")

            # save best by test mIoU (선택)
            if miou > best_miou:
                best_miou = miou
                best_path = os.path.join(run["ckpt_dir"], "best_by_test_miou.pth")
                save_ckpt(best_path, model, optimizer, scheduler, epoch=epoch, best_score=best_miou)
                logger.info(f"Saved best_by_test_miou -> {best_path} (mIoU={best_miou:.4f})")

        # periodic save
        if save_epoch > 0 and (epoch % save_epoch == 0):
            p = os.path.join(run["ckpt_dir"], f"epoch{epoch:04d}.pth")
            save_ckpt(p, model, optimizer, scheduler, epoch=epoch, best_score=best_miou)
            logger.info(f"Saved ckpt -> {p}")

    # final save
    final_path = os.path.join(run["ckpt_dir"], "final.pth")
    save_ckpt(final_path, model, optimizer, scheduler, epoch=epochs, best_score=best_miou)

    total_min = (time.perf_counter() - t0_all) / 60.0
    logger.info(f"Training done. Total time: {total_min:.2f} minutes")
    logger.info(f"Saved final -> {final_path}")
    if best_miou >= 0:
        logger.info(f"Best test mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_pth", type=str, required=True)
    args = ap.parse_args()
    main(args.cfg_pth)
