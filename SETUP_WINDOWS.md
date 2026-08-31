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

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 3. Everything else

```powershell
cd D:\CANDY\Segmentation-4folds
pip install -r requirements.txt
```

## 4. Verify the GPU is actually usable (do NOT skip)

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); x=torch.randn(4096,4096,device='cuda'); print('matmul ok', (x@x).sum().item())"
```

Expected: a torch version like `2.7.x+cu128`, `cuda? True`, `NVIDIA GeForce RTX
5090`, and a finite `matmul ok` number. If `cuda? False` → driver too old. If it
prints True but the matmul throws `no kernel image` → a non-cu128 torch slipped
in; run `pip uninstall torch torchvision -y` and redo step 2.

## 5. Smoke-test the two pipelines

```powershell
# Flood data (small, ~800MB) + a 1-epoch sanity run on the U-Net baseline
python sen1floods11.py --download
python sen1floods11.py --check
python train_flood.py zheng_baseline -e 1 -b 8

# Rainband pipeline (needs cropped_images/ cropped_masks/ present)
python check_pairs.py
```

If both print sane numbers, the machine is ready.
