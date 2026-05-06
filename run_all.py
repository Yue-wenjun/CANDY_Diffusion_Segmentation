"""
Full pipeline runner:
  Phase 1 — Train all comparison + ablation models (4-fold CV)
  Phase 2 — Test each model on the shared held-out clean test set (best fold)
  Phase 3 — Test baseline (best fold) on noisy data at 0 / 10 / 20 dB

Best fold is selected automatically by highest val IoU from training output.
Results are saved to best_folds.json and log.txt.
"""

import subprocess
import datetime
import json
import re
import time

from config import BASE_CONFIG, ABLATION_REGISTRY

# ── Configuration ─────────────────────────────────────────────────────────────

EPOCHS  = BASE_CONFIG["epochs"]
K_FOLDS = BASE_CONFIG["k_folds"]

# Models to train (order matters: train comparison models first)
TRAIN_MODELS = [
    "baseline",
    "segformer",
    "mobilevit",
    "no_skip",
    "simple_cnn",
    "simple_decoder",
    "sde",
    "adjust_steps",
    "ddpm",         # DDPM forward-process baseline (shows CANDY > Gaussian noise)
]

# Noise robustness: only for T=2 baseline, T=5, SegFormer, MobileViT
NOISE_VARIANTS = {
    "baseline":      ["baseline_0dB",      "baseline_10dB",      "baseline_20dB"],
    "adjust_steps":  ["adjust_steps_0dB",  "adjust_steps_10dB",  "adjust_steps_20dB"],
    "segformer":     ["segformer_0dB",     "segformer_10dB",     "segformer_20dB"],
    "mobilevit":     ["mobilevit_0dB",     "mobilevit_10dB",     "mobilevit_20dB"],
    "ddpm":          ["ddpm_0dB",          "ddpm_10dB",          "ddpm_20dB"],
}

BEST_FOLDS_PATH = "best_folds.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(cmd: list[str]) -> str:
    """Run subprocess, stream output to stdout + log.txt, return full stdout."""
    _log(f"CMD: {' '.join(cmd)}")
    lines = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    with open("log.txt", "a", encoding="utf-8") as log_f:
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
            lines.append(line)
    proc.wait()
    return "".join(lines)


def parse_best_fold(output: str) -> int:
    """
    Parse per-fold IoU from main.py output.
    Expects lines like:  Fold 0: Val_Loss=0.123, IoU=0.456, ...
    Returns 1-indexed best fold number, or 1 if not found.
    """
    best_fold, best_iou = 1, -1.0
    for line in output.splitlines():
        m = re.search(r"Fold\s+(\d+).*?IoU=([\d.]+)", line)
        if m:
            fold_idx = int(m.group(1))   # 0-indexed from main.py
            iou = float(m.group(2))
            if iou > best_iou:
                best_iou = iou
                best_fold = fold_idx + 1  # convert to 1-indexed for test.py
    return best_fold


# ── Phase 1: Training ─────────────────────────────────────────────────────────

def phase_train():
    """Train all models; return {model_name: best_fold (1-indexed)}."""
    best_folds = {}

    # Load existing results so a re-run can resume
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

        _log(f"{'='*60}")
        _log(f"TRAIN: {model}")
        _log(f"{'='*60}")

        try:
            output = run_cmd(["python", "main.py", model, "-e", str(EPOCHS), "-k", str(K_FOLDS)])
            bf = parse_best_fold(output)
            best_folds[model] = bf
            _log(f">>> {model}: best fold = {bf}")
        except Exception as e:
            _log(f"ERROR training {model}: {e}")
            best_folds[model] = 1   # fallback

        # Persist after each model so a crash doesn't lose all progress
        with open(BEST_FOLDS_PATH, "w") as f:
            json.dump(best_folds, f, indent=2)

        time.sleep(2)

    return best_folds


# ── Phase 2: Clean test ───────────────────────────────────────────────────────

def phase_test_clean(best_folds: dict[str, int]):
    _log(f"\n{'='*60}")
    _log("PHASE 2: Clean test (held-out 15%)")
    _log(f"{'='*60}")

    for model in TRAIN_MODELS:
        fold = best_folds.get(model, 1)
        _log(f"TEST (clean): {model}  fold={fold}")
        try:
            run_cmd(["python", "test.py", model, "-k", str(K_FOLDS), "-f", str(fold)])
        except Exception as e:
            _log(f"ERROR testing {model}: {e}")
        time.sleep(1)


# ── Phase 3: Noise robustness test ───────────────────────────────────────────

def phase_test_noise(best_folds: dict[str, int]):
    _log(f"\n{'='*60}")
    _log("PHASE 3: Noise robustness test (0 / 10 / 20 dB)")
    _log(f"{'='*60}")

    for base_model, noise_entries in NOISE_VARIANTS.items():
        fold = best_folds.get(base_model, 1)
        for noise_entry in noise_entries:
            if noise_entry not in ABLATION_REGISTRY:
                _log(f"[SKIP] {noise_entry} not in ABLATION_REGISTRY")
                continue
            _log(f"TEST (noise): {noise_entry}  fold={fold}")
            try:
                run_cmd(["python", "test.py", noise_entry, "-k", str(K_FOLDS), "-f", str(fold)])
            except Exception as e:
                _log(f"ERROR testing {noise_entry}: {e}")
            time.sleep(1)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    total_start = datetime.datetime.now()
    _log(f"run_all.py started — {total_start.strftime('%Y-%m-%d %H:%M:%S')}")
    _log(f"Models: {TRAIN_MODELS}")
    _log(f"Epochs: {EPOCHS}  |  K-folds: {K_FOLDS}")

    best_folds = phase_train()
    phase_test_clean(best_folds)
    phase_test_noise(best_folds)

    elapsed = datetime.datetime.now() - total_start
    _log(f"\n{'='*60}")
    _log(f"All done.  Total time: {elapsed}")
    _log(f"Best folds: {best_folds}")
    _log(f"{'='*60}")

    # Print summary table
    print("\n=== Best-fold summary ===")
    for model, fold in best_folds.items():
        print(f"  {model:<20} best fold = {fold}")


if __name__ == "__main__":
    main()
