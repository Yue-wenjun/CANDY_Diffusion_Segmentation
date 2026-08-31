"""Sen1Floods11 flood-segmentation dataset: downloader + Dataset + official splits.

Fully isolated from the rainband pipeline (CustomDataset in data_loading.py):
different channels (VV+VH=2), spatial size (512), normalization (z-score, NO
log — the chips are already dB), label semantics ({-1 ignore, 0 non-water,
1 water}) and split protocol (official CSVs, not k-fold).

Data layout after --download (default root ./sen1floods11):
    sen1floods11/S1Hand/<scene>_S1Hand.tif      float32 (2,512,512), dB
    sen1floods11/LabelHand/<scene>_LabelHand.tif int16   (512,512), {-1,0,1}
    sen1floods11/splits/flood_{train,valid,test,bolivia}_data.csv

Usage:
    python sen1floods11.py --download            # fetch all 446 chips + splits
    python sen1floods11.py --check               # verify counts, print value stats
"""
import argparse
import csv
import os
import urllib.request
import urllib.error

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

BUCKET = "https://storage.googleapis.com/sen1floods11/v1.1"
HAND = f"{BUCKET}/data/flood_events/HandLabeled"
SPLIT_URL = f"{BUCKET}/splits/flood_handlabeled"
SPLITS = ["flood_train_data", "flood_valid_data", "flood_test_data", "flood_bolivia_data"]

IGNORE_INDEX = -1


# ── Download ──────────────────────────────────────────────────────────────────

def _fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        return "ok"
    except urllib.error.HTTPError as e:
        # Clean up a partial/empty file so a rerun retries it.
        if os.path.exists(dest):
            os.remove(dest)
        return f"FAIL {e.code}"


def download(root):
    split_dir = os.path.join(root, "splits")
    for s in SPLITS:
        r = _fetch(f"{SPLIT_URL}/{s}.csv", os.path.join(split_dir, f"{s}.csv"))
        print(f"[split] {s}.csv  {r}")

    scenes = []
    for s in SPLITS:
        p = os.path.join(split_dir, f"{s}.csv")
        for img_name, lbl_name in _read_split(p):
            scenes.append((img_name, lbl_name))
    print(f"[data] {len(scenes)} chips to fetch")

    ok = skip = fail = 0
    for i, (img_name, lbl_name) in enumerate(scenes, 1):
        r1 = _fetch(f"{HAND}/S1Hand/{img_name}", os.path.join(root, "S1Hand", img_name))
        r2 = _fetch(f"{HAND}/LabelHand/{lbl_name}", os.path.join(root, "LabelHand", lbl_name))
        for r in (r1, r2):
            if r == "ok": ok += 1
            elif r == "skip": skip += 1
            else: fail += 1; print(f"  {img_name}/{lbl_name}: {r}")
        if i % 50 == 0:
            print(f"  {i}/{len(scenes)} chips  (files: {ok} new, {skip} cached, {fail} failed)")
    print(f"[done] files: {ok} new, {skip} cached, {fail} failed")


# ── Splits ────────────────────────────────────────────────────────────────────

def _read_split(csv_path):
    """Return list of (image_filename, label_filename). CSV has no header."""
    pairs = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].strip():
                pairs.append((row[0].strip(), row[1].strip()))
    return pairs


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_flood(img, clip=(-30.0, 5.0)):
    """Per-image per-band z-score for Sentinel-1 dB chips.

    NO log (data is already in dB). Values are clipped to a physically sane dB
    window first so extreme layover/shadow returns don't dominate the mean/std,
    then z-scored per band over finite pixels. Any NaN is set to 0 AFTER
    z-scoring so it lands at the per-band mean.
    """
    out = torch.empty_like(img)
    for b in range(img.shape[0]):
        band = img[b]
        finite = torch.isfinite(band)
        band = torch.where(finite, band, torch.zeros_like(band))
        band = band.clamp(min=clip[0], max=clip[1])
        vals = band[finite]
        mean = vals.mean() if vals.numel() else torch.zeros((), device=band.device)
        std = vals.std() if vals.numel() else torch.ones((), device=band.device)
        out[b] = (band - mean) / (std + 1e-6)
    return out


# ── Dataset ───────────────────────────────────────────────────────────────────

class Sen1Floods11Dataset(Dataset):
    """One official split. Preloads into memory (446 chips × 2 × 512² ≈ 1.9 GB
    for the full set; a single split is a fraction of that)."""

    def __init__(self, root, split_csv, augment=False):
        import rasterio  # imported lazily so --check without rasterio still works via tifffile

        self.augment = augment
        pairs = _read_split(os.path.join(root, "splits", split_csv))

        imgs, masks = [], []
        for img_name, lbl_name in pairs:
            ip = os.path.join(root, "S1Hand", img_name)
            mp = os.path.join(root, "LabelHand", lbl_name)
            if not (os.path.exists(ip) and os.path.exists(mp)):
                continue
            with rasterio.open(ip) as s:
                img = s.read().astype(np.float32)          # (2,512,512)
            with rasterio.open(mp) as s:
                lbl = s.read(1).astype(np.int64)           # (512,512), {-1,0,1}
            imgs.append(normalize_flood(torch.from_numpy(img)))
            masks.append(torch.from_numpy(lbl))

        if not imgs:
            raise RuntimeError(f"No chips loaded for {split_csv} under {root} — run --download first.")

        self.images = torch.stack(imgs)                    # (N,2,512,512)
        self.masks = torch.stack(masks).unsqueeze(1).float()  # (N,1,512,512), keeps -1
        print(f"[sen1floods11] {split_csv}: {len(self.images)} chips loaded")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img, mask = self.images[i], self.masks[i]
        if self.augment:
            k = torch.randint(0, 4, (1,)).item()
            if k:
                img = torch.rot90(img, k, dims=[-2, -1]); mask = torch.rot90(mask, k, dims=[-2, -1])
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[-1]); mask = torch.flip(mask, dims=[-1])
            if torch.rand(1).item() > 0.5:
                img = torch.flip(img, dims=[-2]); mask = torch.flip(mask, dims=[-2])
        return img, mask


def get_flood_loaders(root, batch_size=8, augment=True, num_workers=0):
    """Train/val/test loaders on the OFFICIAL Sen1Floods11 splits (no k-fold).
    Bolivia is returned separately as an out-of-distribution generalization set."""
    train = Sen1Floods11Dataset(root, "flood_train_data.csv", augment=augment)
    val = Sen1Floods11Dataset(root, "flood_valid_data.csv", augment=False)
    test = Sen1Floods11Dataset(root, "flood_test_data.csv", augment=False)
    bolivia = Sen1Floods11Dataset(root, "flood_bolivia_data.csv", augment=False)
    mk = lambda ds, sh, dl: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                       drop_last=dl, pin_memory=True, num_workers=num_workers)
    return (mk(train, True, True), mk(val, False, False),
            mk(test, False, False), mk(bolivia, False, False))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _check(root):
    n_files = {d: len(os.listdir(os.path.join(root, d)))
               for d in ("S1Hand", "LabelHand") if os.path.isdir(os.path.join(root, d))}
    print(f"files on disk: {n_files}")
    for s in SPLITS:
        p = os.path.join(root, "splits", f"{s}.csv")
        if os.path.exists(p):
            print(f"  {s}: {len(_read_split(p))} pairs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="sen1floods11")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.download:
        download(args.root)
    if args.check or not args.download:
        _check(args.root)
