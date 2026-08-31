"""Sen1Floods11 flood segmentation — training/eval entry.

Isolated from run_all.py / main.py (the rainband pipeline). Reuses only the
model factory (DiffusionModelWrapper) and checkpoint I/O. Everything protocol-
specific to Sen1Floods11 lives here or in sen1floods11.py:
  - 2-channel VV+VH input, 512×512, official train/valid/test splits (no k-fold)
  - label {-1 ignore, 0 non-water, 1 water}; loss and metrics mask the -1
  - metrics: water-class IoU, both micro (pooled over pixels) and macro
    (per-chip mean, the number the Bonafilia et al. 2020 baseline reports)

Run:
    python train_flood.py zheng_baseline -e 100
    python train_flood.py baseline -e 100 -b 8        # CANDY
"""
import argparse
import os
import time

import torch
import torch.nn as nn

from config import BASE_CONFIG
from models.models import DiffusionModelWrapper
from sen1floods11 import get_flood_loaders, IGNORE_INDEX


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def flood_config(model_type, batch_size):
    """BASE_CONFIG with the Sen1Floods11 protocol overrides applied."""
    cfg = BASE_CONFIG.copy()
    cfg.update({
        "in_channel": 2,          # VV + VH
        "hidden_channel": 2,      # keep CANDY forward chain width == in_channel
        "out_channel": 2,
        "num_classes": 1,         # binary water, sigmoid + BCE
        "input_size": 512,
        "hidden_size": 512,
        "batch_size": batch_size,
        "center_crop": None,      # score the full 512×512 chip (leaderboard protocol)
    })
    if model_type == "segformer":
        cfg["decoder_type"] = "segformer_b0"
    elif model_type == "mobilevit":
        cfg["decoder_type"] = "mobilevit_small"
    return cfg


# ── Masked loss ───────────────────────────────────────────────────────────────

class MaskedBCE(nn.Module):
    """BCE-with-logits over valid pixels only (label != IGNORE_INDEX).

    pos_weight compensates the heavy water/non-water imbalance (water is often
    <5% of valid pixels)."""
    def __init__(self, pos_weight, device):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=torch.tensor([pos_weight], device=device))

    def forward(self, logits, mask):
        valid = (mask != IGNORE_INDEX).float()
        target = mask.clamp(min=0.0)            # -1 -> 0, harmless (masked out anyway)
        loss = self.bce(logits, target) * valid
        denom = valid.sum().clamp(min=1.0)
        return loss.sum() / denom


# ── Ignore-aware metrics ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_flood(model, loader, device, criterion=None, thresh=0.0):
    """Water-class IoU at logit `thresh` (0.0 == prob 0.5), ignoring -1 pixels.

    Returns micro (pooled over all valid pixels) and macro (mean over chips that
    contain water) IoU. Macro is the Sen1Floods11 leaderboard metric."""
    model.eval()
    tp = fp = fn = 0
    per_chip, chip_n = 0.0, 0
    total_loss, nb = 0.0, 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        if criterion is not None:
            total_loss += criterion(logits, masks).item(); nb += 1

        valid = masks != IGNORE_INDEX
        pred = (logits > thresh) & valid
        gt = (masks > 0.5) & valid

        b_tp = (pred & gt).sum(dim=(1, 2, 3))
        b_fp = (pred & ~gt & valid).sum(dim=(1, 2, 3))
        b_fn = (~pred & gt).sum(dim=(1, 2, 3))
        tp += b_tp.sum().item(); fp += b_fp.sum().item(); fn += b_fn.sum().item()

        # per-chip IoU, only for chips that actually contain water (macro metric)
        for i in range(images.size(0)):
            if b_tp[i] + b_fn[i] > 0:
                u = (b_tp[i] + b_fp[i] + b_fn[i]).item()
                per_chip += (b_tp[i].item() / u) if u > 0 else 0.0
                chip_n += 1

    micro = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    macro = per_chip / chip_n if chip_n else float("nan")
    return {
        "iou_micro": micro,
        "iou_macro": macro,
        "loss": total_loss / nb if nb else float("nan"),
        "n_water_chips": chip_n,
    }


# ── Train ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Sen1Floods11 flood segmentation.")
    ap.add_argument("model_type", help="baseline | zheng_baseline | segformer | mobilevit | pure_unet | ...")
    ap.add_argument("-e", "--epochs", type=int, default=100)
    ap.add_argument("-b", "--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos_weight", type=float, default=15.0,
                    help="BCE positive-class weight; ~ (non-water/water) valid-pixel ratio")
    ap.add_argument("--root", default="sen1floods11")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    cfg = flood_config(args.model_type, args.batch_size)
    train_loader, val_loader, test_loader, bolivia_loader = get_flood_loaders(
        args.root, batch_size=args.batch_size, augment=True)

    model = DiffusionModelWrapper(cfg).create_model(args.model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
        epochs=args.epochs, pct_start=0.1)
    criterion = MaskedBCE(args.pos_weight, device)

    ckpt = args.ckpt or f"checkpoint/flood_{args.model_type}_best.pth"
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)

    best_macro = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for bi, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            running += loss.item()
        vm = evaluate_flood(model, val_loader, device, criterion)
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={running/len(train_loader):.4f}  "
              f"val: IoU_macro={vm['iou_macro']:.4f} IoU_micro={vm['iou_micro']:.4f} "
              f"loss={vm['loss']:.4f}  ({time.time()-t0:.0f}s)")

        if vm["iou_macro"] > best_macro:
            best_macro = vm["iou_macro"]
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_iou_macro": best_macro}, ckpt)
            print(f"  best updated (val IoU_macro={best_macro:.4f}) -> {ckpt}")

    # Final test on the best checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
    tm = evaluate_flood(model, test_loader, device)
    bm = evaluate_flood(model, bolivia_loader, device)
    print(f"\n=== {args.model_type} on Sen1Floods11 ===")
    print(f"TEST     : IoU_macro={tm['iou_macro']:.4f}  IoU_micro={tm['iou_micro']:.4f}  "
          f"(n_water_chips={tm['n_water_chips']})")
    print(f"BOLIVIA  : IoU_macro={bm['iou_macro']:.4f}  IoU_micro={bm['iou_micro']:.4f}  "
          f"(out-of-distribution generalization)")
    print("Bonafilia et al. 2020 U-Net baseline reference: test IoU ~0.55-0.62 (macro).")


if __name__ == "__main__":
    main()
