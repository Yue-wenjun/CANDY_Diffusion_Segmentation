"""Hard check that PyTorch is the CUDA (cu128) build and the 5090 actually runs.

Exits non-zero with a precise diagnosis if a CPU-only torch slipped in, so it
can gate a run:  `python verify_gpu.py && python train_flood.py ...`

Why this exists: on Windows, a plain `pip install torch` pulls the CPU-only
PyPI wheel. It imports fine and silently runs everything on the CPU — training
is just 10-50x slower with no error. The only reliable signal is
`torch.version.cuda`: it is a real version string ("12.8") for a CUDA build and
None for a CPU build.
"""
import sys

try:
    import torch
except ImportError:
    sys.exit("FAIL: torch is not installed. See SETUP_WINDOWS.md.")

ver = torch.__version__
cuda_ver = torch.version.cuda          # None on a CPU-only build
print(f"torch.__version__   = {ver}")
print(f"torch.version.cuda  = {cuda_ver}")
print(f"cuda.is_available() = {torch.cuda.is_available()}")

if cuda_ver is None or ver.endswith("+cpu"):
    sys.exit(
        "\nFAIL: this is a CPU-ONLY PyTorch build (would train on CPU, silently, ~10-50x slower).\n"
        "Fix:\n"
        "  pip uninstall torch torchvision -y\n"
        "  pip cache purge\n"
        "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
    )

if not torch.cuda.is_available():
    sys.exit(
        "\nFAIL: CUDA build present but no GPU visible. Usually the NVIDIA driver is too old "
        "for the 5090 — update to a 572+ GeForce driver and check `nvidia-smi` lists the RTX 5090."
    )

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"device              = {name}  (sm_{cap[0]}{cap[1]})")

# Actually execute a kernel — a non-cu128 build on Blackwell imports and reports
# available but throws "no kernel image is available" the moment it runs an op.
try:
    x = torch.randn(4096, 4096, device="cuda")
    val = (x @ x).sum().item()
except RuntimeError as e:
    sys.exit(
        f"\nFAIL: GPU op raised: {e}\n"
        "This is the Blackwell/sm_120 kernel gap — torch is a CUDA build but NOT cu128. Fix:\n"
        "  pip uninstall torch torchvision -y && pip cache purge\n"
        "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
    )

if cap[0] < 12:
    print(f"WARN: expected sm_120 for an RTX 5090, got sm_{cap[0]}{cap[1]} — different GPU?")

print(f"\nPASS: GPU compute works (matmul sum={val:.1f}). torch is {ver}, ready to train.")
