import os
import argparse
import sys

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from util.util import load_yaml, set_seed, setup_logger, ensure_dir, Timer, save_checkpoint
from util.dataset import build_dataset, PKBatchSampler
from util.model import TimmOstClassifier
from util.metric import confusion_matrix, classification_metrics_from_cm
from util.loss import ce_loss, triplet_loss_all, triplet_loss_hard


@torch.no_grad()
def evaluate(model, loader, device, num_class):
    if loader is None or len(loader.dataset) == 0:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    model.eval()
    all_pred, all_y = [], []

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


def build_loss_string(use_ce, use_tri, avg_ce, avg_tri, avg_total):
    """
    loss 구성에 따라 출력 문자열을 다르게 생성:
      - CE only:        ce=...
      - Triplet only:   tri=...
      - CE+Triplet:     ce=... | tri=... | total=...
    """
    if use_ce and use_tri:
        return f"ce={avg_ce:.6f} | tri={avg_tri:.6f} | total={avg_total:.6f}"
    elif use_ce:
        return f"ce={avg_ce:.6f}"
    elif use_tri:
        return f"tri={avg_tri:.6f}"
    else:
        return f"total={avg_total:.6f}"


def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    out_dir = cfg["train"]["out_dir"]
    ensure_dir(out_dir)
    logger = setup_logger(out_dir, name="train", filename="log.txt")

    set_seed(int(cfg["seed"]))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    num_class = int(cfg["task"]["num_class"])
    class_names = cfg["task"]["class_names_2"] if num_class == 2 else cfg["task"]["class_names_3"]

    # dataset
    train_ds, _ = build_dataset(cfg, split="train")
    test_ds, _ = build_dataset(cfg, split="test")

    # loader
    num_workers = int(cfg["train"]["num_workers"])
    batch_sampler_cfg = cfg["train"]["batch_sampler"]

    if bool(batch_sampler_cfg["enabled"]):
        n_class = int(batch_sampler_cfg["n_class"])
        n_samples = int(batch_sampler_cfg["n_samples"])

        bsampler = PKBatchSampler(
            class_to_indices=train_ds.class_to_indices,
            n_class=n_class,
            n_samples=n_samples,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=bsampler,
            num_workers=num_workers,
            pin_memory=True
        )

        effective_bs = n_class * n_samples
        bs_desc = f"{n_class}*{n_samples} (= {effective_bs})"
    else:
        effective_bs = int(batch_sampler_cfg.get("batch_size", cfg["train"].get("batch_size", 64)))
        bs_desc = f"{effective_bs}"
        train_loader = DataLoader(
            train_ds,
            batch_size=effective_bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )


    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    train_eval_loader = DataLoader(
            train_ds,
            batch_size=64,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    # model
    mcfg = cfg["model"]
    model = TimmOstClassifier(
        backbone=mcfg["backbone"],
        pretrained=bool(mcfg["pretrained"]),
        embed_dim=int(mcfg["embed_dim"]),
        num_class=num_class,
    ).to(device)

    # optim
    ocfg = cfg["optim"]
    lr = float(ocfg["lr"])
    wd = float(ocfg["weight_decay"])
    if ocfg["name"].lower() == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif ocfg["name"].lower() == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optim: {ocfg['name']}")

    # amp
    use_fp16 = bool(cfg["train"]["fp16"])
    scaler = GradScaler(enabled=use_fp16)

    # loss cfg
    lcfg = cfg["loss"]
    use_ce = bool(lcfg["use_ce"])
    ce_w = float(lcfg["ce_weight"])
    use_tri = bool(lcfg["use_triplet"])
    tri_w = float(lcfg["triplet_weight"])
    tri_margin = float(lcfg["triplet_margin"])
    tri_mining = str(lcfg["triplet_mining"]).lower()

    epochs = int(cfg["train"]["epochs"])
    log_epoch = int(cfg["train"].get("log_epoch", 1))
    eval_freq = int(cfg["train"]["eval_freq"])
    save_epoch = int(cfg["train"]["save_epoch"])

    # logging header
    logger.info("======== Experiment Setup ========")
    logger.info(f"Config: {cfg_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Num class: {num_class} / class_names={class_names}")
    logger.info(f"Train samples: {len(train_ds)}")
    logger.info(f"Test samples: {len(test_ds)}")
    logger.info(f"Backbone: {mcfg['backbone']} (pretrained={mcfg['pretrained']})")
    logger.info(f"Embed dim: {mcfg['embed_dim']}")
    logger.info(f"Epochs: {epochs}, lr={lr}, wd={wd}")
    logger.info(f"Batch sampler: {bool(batch_sampler_cfg['enabled'])}, batch_size={bs_desc}")
    logger.info(f"FP16: {use_fp16}")
    logger.info(f"Loss: CE({use_ce}, w={ce_w}) + Triplet({use_tri}, w={tri_w}, margin={tri_margin}, mining={tri_mining})")
    logger.info(f"log_epoch={log_epoch}, eval_freq={eval_freq}, save_epoch={save_epoch}")
    logger.info("==================================")

    best_acc = -1.0
    total_timer = Timer()
    total_timer.start()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_timer = Timer()
        ep_timer.start()

        running_total = 0.0
        running_ce = 0.0
        running_tri = 0.0
        steps = 0

        for x, y, _path in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with autocast(enabled=use_fp16):
                emb, logits = model(x)

                ce_val = emb.new_tensor(0.0)
                tri_val = emb.new_tensor(0.0)

                if use_ce:
                    ce_val = ce_loss(logits, y)  # unweighted

                if use_tri:
                    if tri_mining == "hard":
                        tri_val = triplet_loss_hard(emb, y, tri_margin)  # unweighted
                    else:
                        tri_val = triplet_loss_all(emb, y, tri_margin)   # unweighted

                # weighted total
                loss = emb.new_tensor(0.0)
                if use_ce:
                    loss = loss + ce_w * ce_val
                if use_tri:
                    loss = loss + tri_w * tri_val

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running_total += float(loss.item())
            if use_ce:
                running_ce += float(ce_val.item())
            if use_tri:
                running_tri += float(tri_val.item())
            steps += 1

        if steps == 0:
            logger.error("No training steps executed. Check PKBatchSampler settings (n_class/n_samples) and drop_last.")
            raise RuntimeError("No training steps executed.")

        avg_total = running_total / steps
        avg_ce = (running_ce / steps) if use_ce else 0.0
        avg_tri = (running_tri / steps) if use_tri else 0.0
        loss_str = build_loss_string(use_ce, use_tri, avg_ce, avg_tri, avg_total)

        if (epoch % eval_freq) == 0:
            eval_timer = Timer()
            eval_timer.start()
            
            train_m = evaluate(model, train_eval_loader, device, num_class)
            test_m = evaluate(model, test_loader, device, num_class)

            eval_time = eval_timer.elapsed()
            ep_time = ep_timer.elapsed()

            logger.info(
                f"[Epoch {epoch:04d}/{epochs}] "
                f"{loss_str} | "
                f"train(acc={train_m['acc']:.4f}, p={train_m['precision']:.4f}, r={train_m['recall']:.4f}, f1={train_m['f1']:.4f}) | "
                f"test(acc={test_m['acc']:.4f}, p={test_m['precision']:.4f}, r={test_m['recall']:.4f}, f1={test_m['f1']:.4f}) | "
                f"epoch_time={ep_time:.1f}s eval_time={eval_time:.1f}s"
            )

            # best by acc
            if test_m["acc"] > best_acc:
                best_acc = test_m["acc"]
                save_checkpoint(
                    os.path.join(out_dir, "best.pth"),
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "scaler": scaler.state_dict(),
                        "best_acc": best_acc,
                        "cfg": cfg,
                    }
                )
        else:
            ep_time = ep_timer.elapsed()
            if (epoch % log_epoch) == 0:
                logger.info(
                    f"[Epoch {epoch:04d}/{epochs}] {loss_str} | epoch_time={ep_time:.1f}s"
                )

        # periodic ckpt
        if (epoch % save_epoch) == 0:
            save_checkpoint(
                os.path.join(out_dir, f"epoch{epoch:04d}.pth"),
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_acc": best_acc,
                    "cfg": cfg,
                }
            )

    total_time = total_timer.elapsed()
    logger.info(f"Training done. Total time: {total_time/60:.2f} minutes")
    logger.info(f"Best test acc: {best_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train OST classifier")
    parser.add_argument(
        "--cfg_pth",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    if any(a.startswith("--cfg_pth") for a in sys.argv[1:]):
        args = parse_args()
        cfg_path = args.cfg_pth
    else:
        cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    main(cfg_path)
