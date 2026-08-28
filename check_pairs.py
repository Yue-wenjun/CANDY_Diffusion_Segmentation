"""Diagnose image/mask tile pairing mismatches (read-only, changes nothing).

Tiles are named <scene>_pstrips_<N>.tif by data_processing.py. Pairing in
data_loading.py is positional over the two sorted listings, so ANY count
difference misaligns everything after it. This script groups tiles by scene,
zips the sorted scene lists (image scene stems and mask scene stems may
differ), and reports every scene whose tile counts disagree, with the exact
tile numbers present on one side only.

Usage: python check_pairs.py [image_dir] [mask_dir]
"""
import os
import sys
from collections import defaultdict

img_dir = sys.argv[1] if len(sys.argv) > 1 else "cropped_images"
mask_dir = sys.argv[2] if len(sys.argv) > 2 else "cropped_masks"


def index_by_scene(d):
    by_scene = defaultdict(set)
    for f in os.listdir(d):
        stem = os.path.splitext(f)[0]
        if "_pstrips_" in stem:
            scene, n = stem.rsplit("_pstrips_", 1)
            try:
                by_scene[scene].add(int(n))
            except ValueError:
                by_scene[stem].add(-1)   # unexpected suffix, surface as its own scene
        else:
            by_scene[stem].add(-1)       # unexpected naming, surface it
    return by_scene


imgs, masks = index_by_scene(img_dir), index_by_scene(mask_dir)
iscenes, mscenes = sorted(imgs), sorted(masks)
print(f"{img_dir}:  {sum(len(v) for v in imgs.values())} tiles in {len(iscenes)} scenes")
print(f"{mask_dir}: {sum(len(v) for v in masks.values())} tiles in {len(mscenes)} scenes")
if len(iscenes) != len(mscenes):
    print(f"!! scene count differs: {len(iscenes)} image scenes vs {len(mscenes)} mask scenes")

bad = 0
for iscene, mscene in zip(iscenes, mscenes):
    ic, mc = imgs[iscene], masks[mscene]
    if len(ic) != len(mc):
        bad += 1
        print(f"\nMISMATCH  image scene '{iscene}' ({len(ic)} tiles)  <->  "
              f"mask scene '{mscene}' ({len(mc)} tiles)")
        mask_only = sorted(mc - ic)
        img_only = sorted(ic - mc)
        if mask_only:
            print(f"  mask-only tile numbers ({len(mask_only)}): {mask_only[:20]}")
            for n in mask_only[:20]:
                print(f"    -> {os.path.join(mask_dir, f'{mscene}_pstrips_{n}.tif')}")
        if img_only:
            print(f"  image-only tile numbers ({len(img_only)}): {img_only[:20]}")
            for n in img_only[:20]:
                print(f"    -> {os.path.join(img_dir, f'{iscene}_pstrips_{n}.tif')}")

for extra in iscenes[len(mscenes):]:
    bad += 1
    print(f"\nimage-only scene: '{extra}' ({len(imgs[extra])} tiles)")
for extra in mscenes[len(iscenes):]:
    bad += 1
    print(f"\nmask-only scene: '{extra}' ({len(masks[extra])} tiles)")

if bad:
    print(f"\n{bad} mismatched scene(s). Quarantine the extra files (move, don't delete), "
          f"then re-run this check until counts agree.")
else:
    print("\nAll scenes pair up with equal tile counts.")