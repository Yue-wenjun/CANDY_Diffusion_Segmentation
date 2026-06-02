"""
Full pipeline:
  Phase 1 — Train all models (4-fold CV, via subprocess)
  Phase 2 — Test all models × all folds on clean + 0/10/20 dB data (in-process)
             Each dataset loaded once; all models share the same DataLoader.

Results  → noise_test_results/results.csv  (append-safe, resumes on re-run)
Training → best_folds.json + log.txt
"""

import subprocess
import datetime
import json
import re
import time
import os
import torch
import csv

from config import BASE_CONFIG, get_config
from models.models import DiffusionModelWrapper
from data_loading import get_test_only_dataloader
from train import load_checkpoint
from utils import app

# ── Configuration ─────────────────────────────────────────────────────────────

EPOCHS  = BASE_CONFIG["epochs"]
K_FOLDS = BASE_CONFIG["k_folds"]

TRAIN_MODELS = [
    "baseline", "segformer", "mobilevit", "no_skip",
    "simple_cnn", "simple_decoder", "sde", "adjust_steps", "ddpm",
]

# Noise robustness: only these models
NOISE_TEST_MODELS = ["baseline", "adjust_steps", "segformer", "mobilevit", "ddpm"]

NOISE_LEVELS     = [0, 5, 10, 15, 20, 25, 30]
CLEAN_IMAGE_PATH = "cropped_images"
NOISE_PATH_TPL   = "cropped_noised_data/cropped_noised_{db}dB"
MASK_PATH        = "cropped_masks"
CHECKPOINT_TPL   = "checkpoint/{model}_fold{fold}_best.pth"
ADJUST_STEPS_VAL = 5

OUTPUT_ROOT     = "noise_test_results"
CSV_FILENAME    = "results.csv"
CSV_FIELDS      = ["Model", "Fold", "Condition", "Loss", "IoU", "Dice", "Proportion"]
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
    proc  = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
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


# ── Phase 1: Training ─────────────────────────────────────────────────────────

def phase_train():
    best_folds = {}
    try:
        with open(BEST_FOLDS_PATH) as f:
            best_folds = json.load(f)
        _log(f"Loaded existing best_folds: {best_folds}")
    except FileNotFoundError:
        pass

    for model in TRAIN_MODELS:
        if model in best_folds:
            _log(f"[SKIP] {model} already trained (best fold={best_folds[model]})")
            continue

        _log(f"{'='*60}\nTRAIN: {model}\n{'='*60}")
        try:
            output = run_cmd(["python", "main.py", model, "-e", str(EPOCHS), "-k", str(K_FOLDS)])
            bf = parse_best_fold(output)
            best_folds[model] = bf
            _log(f">>> {model}: best fold = {bf}")
        except Exception as e:
            _log(f"ERROR training {model}: {e}")
            best_folds[model] = 1

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
            and os.path.exists(CHECKPOINT_TPL.format(model=m, fold=f))
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

                ckpt = CHECKPOINT_TPL.format(model=model_type, fold=fold)
                if not os.path.exists(ckpt):
                    _log(f"[SKIP] no checkpoint: {ckpt}")
                    continue

                _log(f"TEST: {model_type}  fold={fold}  @ {condition_name}")
                model = DiffusionModelWrapper(config).create_model(model_type).to(device)
                load_checkpoint(model, None, None, ckpt)
                model.eval()

                save_dir = os.path.join(OUTPUT_ROOT, model_type, f"fold{fold}_{condition_name}")
                os.makedirs(save_dir, exist_ok=True)
                metrics = app(model, test_loader, device, BASE_CONFIG["batch_size"], save_dir)

                del model
                torch.cuda.empty_cache()

                if metrics:
                    results.append({
                        "Model":      model_type,
                        "Fold":       fold,
                        "Condition":  condition_name,
                        "Loss":       metrics["loss"],
                        "IoU":        metrics["iou"],
                        "Dice":       metrics["dice"],
                        "Proportion": metrics["proportion"],
                    })
                    done_keys.add(key)
                    _save_csv(results)
                    _log(f"CSV updated ({len(results)} rows)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    t0 = datetime.datetime.now()
    _log(f"run_all.py started — {t0:%Y-%m-%d %H:%M:%S}")
    _log(f"Models: {TRAIN_MODELS}  Epochs: {EPOCHS}  K-folds: {K_FOLDS}")

    best_folds = phase_train()
    phase_test()

    elapsed = datetime.datetime.now() - t0
    _log(f"\n{'='*60}\nAll done. Total: {elapsed}\n{'='*60}")

    print("\n=== Best-fold summary ===")
    for model, fold in best_folds.items():
        print(f"  {model:<20} best fold = {fold}")


if __name__ == "__main__":
    main()
