"""
Full pipeline:
  Phase 1 — Train all models (4-fold CV, via subprocess)
  Phase 2 — Test all models × all folds on clean + 0/10/20 dB data (in-process)
             Each dataset loaded once; all models share the same DataLoader.

Results  → noise_test_results/results.csv  (append-safe, resumes on re-run)
Training → best_folds.json + log.txt
"""

import argparse
import subprocess
import datetime
import json
import re
import time
import os
import shutil
import torch
import csv

from config import BASE_CONFIG, get_config
from models.models import DiffusionModelWrapper
from data_loading import get_test_only_dataloader
from utils import app

# ── Configuration ─────────────────────────────────────────────────────────────

EPOCHS  = 70
K_FOLDS = BASE_CONFIG["k_folds"]

# Loss for Phase 1 training. Passed to main.py via -l.
#   "bce:25.2"     → measured pos_weight = bg/fg over cropped_masks (fg = 3.81%).
#                    Replaces the old guessed 10, which only compensated ~9% fg and
#                    left the decision boundary stuck at logit ~-2.
#                    NOTE: masks also contain a stray value 4.0 (treated as
#                    background). If inspect_mask_values.py says 4.0 is foreground,
#                    use the "4.0 as foreground" pos_weight it prints instead.
#   "dice"         → imbalance-robust but was unstable in earlier runs.
LOSS_SPEC = "bce:25.2"

TRAIN_MODELS = [
    # CANDY + decoder variants
    "baseline", "baseline_tanh",
    "segformer", "mobilevit",
    # Pure decoder baselines (no CANDY)
    "pure_unet", "pure_segformer", "pure_mobilevit",
    # Structural ablations
    "no_skip", "simple_cnn", "simple_decoder", "sde", "adjust_steps", "ddpm",
    # Literature baseline: Zheng et al. 2024 U-Net (loss/scheduler forced by config,
    # the -l flag passed at train time is overridden with a printed note)
    "zheng_baseline",
]

# Noise robustness. Includes the plain-U-Net family (pure_unet, zheng_baseline)
# so CANDY's "diffusion forward chain is noise-robust" claim is tested against a
# like-for-like baseline under identical noise — previously pure_unet was absent,
# so the one place CANDY might win couldn't be measured.
NOISE_TEST_MODELS = ["baseline", "baseline_tanh", "pure_unet", "zheng_baseline",
                     "adjust_steps", "segformer", "mobilevit", "ddpm"]

NOISE_LEVELS     = [0, 10, 20]
CLEAN_IMAGE_PATH = "cropped_images"
NOISE_PATH_TPL   = "cropped_noised_data/cropped_noised_{db}dB"
MASK_PATH        = "cropped_masks"
ADJUST_STEPS_VAL = 5

OUTPUT_ROOT     = "noise_test_results"
CSV_FILENAME    = "results.csv"
# IoU/Dice are foreground-only (empty-GT excluded), at the val-frozen threshold.
# IoU_BestOnTest is the leaky best-on-test ceiling, kept only as a diagnostic.
CSV_FIELDS      = ["Model", "Fold", "Condition", "Loss", "IoU", "Dice", "Pooled_IoU_05",
                   "Proportion", "Best_Thresh", "IoU_BestOnTest", "Empty_FP_Rate"]
BEST_FOLDS_PATH = "best_folds.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(msg):
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(cmd):
    _log(f"CMD: {' '.join(cmd)}")
    lines = []
    # PYTHONUTF8=1 forces the child's stdout to UTF-8 regardless of the Windows
    # console codepage (GBK), so a stray non-GBK character in any print() can
    # never crash a training run again.
    proc  = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )
    with open("log.txt", "a", encoding="utf-8") as log_f:
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
            lines.append(line)
    proc.wait()
    return "".join(lines)


def parse_best_fold(output):
    best_fold, best_iou = 1, -1.0
    for line in output.splitlines():
        m = re.search(r"Fold\s+(\d+).*?IoU=([\d.]+)", line)
        if m:
            fold_idx, iou = int(m.group(1)), float(m.group(2))
            if iou > best_iou:
                best_iou, best_fold = iou, fold_idx + 1
    return best_fold


def _ckpt_path(model, fold):
    step_suffix = f"_T{ADJUST_STEPS_VAL}" if model == "adjust_steps" else ""
    return f"checkpoint/{model}{step_suffix}_fold{fold}_best.pth"


def _csv_path():
    return os.path.join(OUTPUT_ROOT, CSV_FILENAME)


def _load_csv():
    rows, done = [], set()
    if not os.path.exists(_csv_path()):
        return rows, done
    with open(_csv_path(), newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            done.add((row["Model"], int(row["Fold"]), row["Condition"]))
    _log(f"CSV: {len(rows)} existing rows, resuming.")
    return rows, done


def _save_csv(results):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(_csv_path(), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(results)


def archive_stale_artifacts():
    """Move prior-run artifacts aside so training starts from a clean slate.

    Renames (never deletes) into _archive_<timestamp>/ so the old BCE run stays
    recoverable. This is what makes --fresh safe: with the old checkpoints gone,
    run_pipeline can't silently RESUME them (which overran OneCycleLR and, worse,
    would have kept training the old weights instead of the new loss).
    """
    stamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = f"_archive_{stamp}"
    targets = [BEST_FOLDS_PATH, "checkpoint", _csv_path()]

    moved = []
    for t in targets:
        if os.path.exists(t):
            os.makedirs(archive, exist_ok=True)
            dest = os.path.join(archive, os.path.basename(os.path.normpath(t)))
            shutil.move(t, dest)
            moved.append((t, dest))

    if moved:
        for src, dest in moved:
            _log(f"[FRESH] archived {src}  ->  {dest}")
    else:
        _log("[FRESH] nothing to archive (already clean)")


# ── Phase 1: Training ─────────────────────────────────────────────────────────

def _load_best_folds():
    try:
        with open(BEST_FOLDS_PATH) as f:
            best_folds = json.load(f)
        _log(f"Loaded existing best_folds: {best_folds}")
        return best_folds
    except FileNotFoundError:
        return {}


def phase_train():
    best_folds = _load_best_folds()

    for model in TRAIN_MODELS:
        if model in best_folds:
            _log(f"[SKIP] {model} already trained (best fold={best_folds[model]})")
            continue

        _log(f"{'='*60}\nTRAIN: {model}\n{'='*60}")
        try:
            output = run_cmd(["python", "main.py", model, "-e", str(EPOCHS),
                              "-k", str(K_FOLDS), "-l", LOSS_SPEC])
            bf = parse_best_fold(output)
            # main.py swallows per-fold exceptions and exits 0, so a fully
            # failed run still reaches here. Only a produced checkpoint proves
            # training happened — never mark a model trained without one.
            if not os.path.exists(_ckpt_path(model, bf)):
                _log(f"ERROR: {model}: no checkpoint at {_ckpt_path(model, bf)} — "
                     f"training failed, NOT marked as trained. Fix the error and rerun.")
                continue
            best_folds[model] = bf
            _log(f">>> {model}: best fold = {bf}")
        except Exception as e:
            _log(f"ERROR training {model}: {e} — NOT marked as trained.")
            continue

        with open(BEST_FOLDS_PATH, "w") as f:
            json.dump(best_folds, f, indent=2)
        time.sleep(2)

    return best_folds


# ── Phase 2+3: Testing (in-process, each dataset loaded once) ─────────────────

def phase_test():
    device  = torch.device("cuda")
    results, done_keys = _load_csv()

    # Clean test: all models × all folds
    # Noise test: NOISE_TEST_MODELS only × all folds
    conditions = [("Clean", CLEAN_IMAGE_PATH, TRAIN_MODELS)]
    for db in NOISE_LEVELS:
        conditions.append((f"{db}dB", NOISE_PATH_TPL.format(db=db), NOISE_TEST_MODELS))

    _log(f"\n{'='*60}\nPHASE 2+3: Testing\n{'='*60}")

    for condition_name, img_dir, models_for_cond in conditions:
        if not os.path.isdir(img_dir):
            _log(f"[SKIP] {img_dir} not found")
            continue

        pending = [
            (m, f) for m in models_for_cond
            for f in range(1, K_FOLDS + 1)
            if (m, f, condition_name) not in done_keys
            and os.path.exists(_ckpt_path(m, f))
        ]
        if not pending:
            _log(f"[SKIP] {condition_name}: all done")
            continue

        _log(f"Loading: {condition_name}  ({img_dir})")
        test_loader = get_test_only_dataloader(img_dir, MASK_PATH, BASE_CONFIG["batch_size"])
        if len(test_loader.dataset) == 0:
            _log(f"[SKIP] empty: {img_dir}")
            continue

        for model_type in models_for_cond:
            custom_steps = ADJUST_STEPS_VAL if model_type == "adjust_steps" else None
            config, _, _ = get_config(model_type, custom_steps)

            for fold in range(1, K_FOLDS + 1):
                key = (model_type, fold, condition_name)
                if key in done_keys:
                    continue

                ckpt = _ckpt_path(model_type, fold)
                if not os.path.exists(ckpt):
                    _log(f"[SKIP] no checkpoint: {ckpt}")
                    continue

                _log(f"TEST: {model_type}  fold={fold}  @ {condition_name}")
                model = DiffusionModelWrapper(config).create_model(model_type).to(device)
                # Load weights + the val-frozen decision threshold in one read.
                ckpt_data  = torch.load(ckpt, map_location=device)
                model.load_state_dict(ckpt_data["model_state_dict"])
                frozen_t   = ckpt_data.get("best_thresh")   # None for legacy checkpoints
                model.eval()

                save_dir = os.path.join(OUTPUT_ROOT, model_type, f"fold{fold}_{condition_name}")
                os.makedirs(save_dir, exist_ok=True)
                metrics = app(model, test_loader, device, BASE_CONFIG["batch_size"], save_dir,
                              thresh=frozen_t, center_crop=config.get("center_crop"))

                del model
                torch.cuda.empty_cache()

                if metrics:
                    results.append({
                        "Model":      model_type,
                        "Fold":       fold,
                        "Condition":  condition_name,
                        "Loss":          metrics["loss"],
                        "IoU":           metrics["iou"],
                        "Dice":          metrics["dice"],
                        "Pooled_IoU_05": metrics.get("pooled_iou_05"),
                        "Proportion":    metrics["proportion"],
                        "Best_Thresh":    metrics["best_thresh"],
                        "IoU_BestOnTest": metrics["iou_best_on_test"],
                        "Empty_FP_Rate":  metrics["empty_fp_rate"],
                    })
                    done_keys.add(key)
                    _save_csv(results)
                    _log(f"CSV updated ({len(results)} rows)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: Phase 1 (train) + Phase 2/3 (test).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip Phase 1 entirely; run testing only against existing checkpoints.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Archive prior best_folds/checkpoints/results into _archive_<ts>/ and "
             "retrain everything from scratch. Use this when changing the loss (e.g. "
             "bce -> bce:25.2) so nothing resumes stale weights.",
    )
    args = parser.parse_args()

    if args.fresh and args.skip_train:
        parser.error("--fresh and --skip-train are mutually exclusive.")

    t0 = datetime.datetime.now()
    _log(f"run_all.py started — {t0:%Y-%m-%d %H:%M:%S}")
    _log(f"Models: {TRAIN_MODELS}  Epochs: {EPOCHS}  K-folds: {K_FOLDS}  Loss: {LOSS_SPEC}")

    if args.skip_train:
        _log("[SKIP] Phase 1 — testing only (--skip-train)")
        best_folds = _load_best_folds()
    else:
        if args.fresh:
            _log(f"[FRESH] clean-slate retrain with loss '{LOSS_SPEC}'")
            archive_stale_artifacts()
        best_folds = phase_train()
    phase_test()

    elapsed = datetime.datetime.now() - t0
    _log(f"\n{'='*60}\nAll done. Total: {elapsed}\n{'='*60}")

    print("\n=== Best-fold summary ===")
    for model, fold in best_folds.items():
        print(f"  {model:<20} best fold = {fold}")


if __name__ == "__main__":
    main()
