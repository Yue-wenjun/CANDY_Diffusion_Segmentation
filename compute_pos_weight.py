"""
Compute BCEWithLogitsLoss pos_weight from the mask set.

pos_weight = (#background pixels) / (#foreground pixels)

This is the value that makes the positive (foreground) term of BCE carry the
same total weight as the negative (background) term, i.e. it exactly cancels the
class imbalance. The current hard-coded 10 in main.py was a guess; run this to
replace it with the data-driven number.

Mask reading mirrors data_loading.py exactly (rasterio + binarize on == 1),
so the count reflects what the model actually trains on.

Usage:
    python compute_pos_weight.py                 # defaults to cropped_masks
    python compute_pos_weight.py <mask_dir>
"""

import os
import sys
import numpy as np

try:
    import rasterio
except ImportError:
    sys.exit("rasterio not installed — run inside the same env as training (candy_env).")


def binarize(mask):
    # Identical to data_loading.CustomDataset: foreground is strictly value == 1,
    # everything else (incl. NaN) becomes 0.
    mask = np.where(mask == 1, 1, mask)
    mask = np.where(mask != 1, 0, mask)
    return np.nan_to_num(mask, nan=0.0).astype(np.float64)


def main():
    mask_dir = sys.argv[1] if len(sys.argv) > 1 else "cropped_masks"
    if not os.path.isdir(mask_dir):
        sys.exit(f"mask dir not found: {mask_dir}")

    files = sorted(os.listdir(mask_dir))
    if not files:
        sys.exit(f"no files in {mask_dir}")

    fg_pixels = 0          # count of value-1 pixels
    total_pixels = 0
    empty_masks = 0        # images with zero foreground
    n = 0
    raw_values = set()     # sanity: what values actually appear (pre-binarize)

    print(f"Scanning {len(files)} masks in '{mask_dir}' ...")
    for i, fname in enumerate(files):
        path = os.path.join(mask_dir, fname)
        try:
            with rasterio.open(path) as src:
                raw = src.read()          # [bands, H, W]
        except Exception as e:
            print(f"  [skip] {fname}: {e}")
            continue

        # Record a small sample of raw values to verify the '== 1' assumption.
        if len(raw_values) < 20:
            raw_values.update(np.unique(raw).tolist()[:20])

        m = binarize(raw)
        fg = int(m.sum())
        fg_pixels += fg
        total_pixels += m.size
        if fg == 0:
            empty_masks += 1
        n += 1

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(files)}  (running fg fraction: "
                  f"{fg_pixels / total_pixels:.5f})")

    if fg_pixels == 0:
        print("\n[!!] ZERO foreground pixels found across the whole set.")
        print("     The '== 1' binarization is almost certainly wrong for these masks.")
        print(f"     Raw values seen (sample): {sorted(raw_values)}")
        print("     Fix data_loading.py's binarization before trusting any training.")
        return

    bg_pixels = total_pixels - fg_pixels
    fg_fraction = fg_pixels / total_pixels
    pos_weight = bg_pixels / fg_pixels

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"masks scanned            : {n}")
    print(f"raw values seen (sample) : {sorted(raw_values)}   (foreground assumed == 1)")
    print(f"foreground pixels        : {fg_pixels:,}")
    print(f"background pixels        : {bg_pixels:,}")
    print(f"foreground fraction      : {fg_fraction:.5f}  ({fg_fraction * 100:.3f}%)")
    print(f"empty masks (no fg)      : {empty_masks} / {n}  ({empty_masks / n:.4f})")
    print("-" * 60)
    print(f">>> pos_weight (bg/fg)   : {pos_weight:.2f}")
    print(f">>> sqrt(pos_weight)     : {pos_weight ** 0.5:.2f}   (milder, if full value destabilizes)")
    print("-" * 60)
    print("Plug into run_all.py:")
    print(f'    LOSS_SPEC = "bce:{pos_weight:.1f}"')
    print("or run a single model directly:")
    print(f"    python main.py baseline -e 70 -k 4 -l bce:{pos_weight:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()