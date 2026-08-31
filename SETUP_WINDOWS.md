# Windows setup for RTX 5090 (Blackwell) — from scratch

The one thing that breaks everything: the 5090 is **Blackwell (compute
capability sm_120)**, and only **CUDA 12.8** PyTorch wheels (`cu128`) contain
kernels for it. Install a `cu121`/`cu124`/CPU build and every GPU op dies with
`no kernel image is available for execution on the device`.

You do **not** need to install the CUDA Toolkit — the `cu128` wheels bundle the
CUDA runtime. You only need a **recent NVIDIA GeForce driver** (572+, newer is
fine). Get it from GeForce Experience or nvidia.com, then `nvidia-smi` should
list the RTX 5090.

## 1. Python env (Miniconda)

Install Miniconda (https://docs.conda.io/en/latest/miniconda.html), then in an
**Anaconda PowerShell Prompt**:

```powershell
conda create -n candy python=3.11 -y
conda activate candy
python -m pip install --upgrade pip
```

## 2. PyTorch — cu128, BEFORE anything else

**The #1 trap: a plain `pip install torch` on Windows installs the CPU-only
wheel.** It imports fine and trains — just silently on the CPU, ~10-50x slower,
no error. Three rules to never hit it:

1. **Always** use `--index-url` (note: `--index-url`, NOT `--extra-index-url` —
   the former searches ONLY the cu128 index so a CPU wheel is impossible; the
   latter can still resolve to the CPU wheel from PyPI).
2. Install torch **first**, before any package that depends on it, so pip can't
   pull a CPU torch in as a side-dependency.
3. Never put `torch` in requirements.txt (it is deliberately absent).

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 3. Everything else

```powershell
cd D:\CANDY\Segmentation-4folds
pip install -r requirements.txt
```

## 4. Verify the GPU is actually usable (do NOT skip)

A committed hard-check that fails loudly on a CPU build, a too-old driver, or the
Blackwell kernel gap:

```powershell
python verify_gpu.py
```

Must end with `PASS: GPU compute works ...` and show `torch.version.cuda = 12.8`,
`cuda.is_available() = True`, `NVIDIA GeForce RTX 5090 (sm_120)`. Any `FAIL:`
line prints the exact fix. The tell-tale of a CPU build is
`torch.version.cuda = None` (or a `+cpu` version string).

**Re-run `python verify_gpu.py` after step 3 as well** — installing other
packages can silently downgrade or replace torch with a CPU build. If it ever
regresses:

```powershell
pip uninstall torch torchvision -y
pip cache purge          # so pip does not reinstall the cached CPU wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 5. Smoke-test the two pipelines

```powershell
# Flood data (small, ~800MB) + a 1-epoch sanity run on the U-Net baseline.
# verify_gpu.py gates the run so training can never start on a CPU torch.
python verify_gpu.py
python sen1floods11.py --download
python sen1floods11.py --check
python verify_gpu.py; if ($?) { python train_flood.py zheng_baseline -e 1 -b 8 }

# Rainband pipeline (needs cropped_images/ cropped_masks/ present)
python check_pairs.py
```

If both print sane numbers, the machine is ready.
