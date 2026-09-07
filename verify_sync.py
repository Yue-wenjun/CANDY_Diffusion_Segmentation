"""Fingerprint the source tree so a manual (no-network) sync can be verified.

The D: training machine has no network; code is copied by hand, which has
repeatedly caused truncated files and mixed-version runs (a new train.py next to
an old run_all.py, etc.). Run this on BOTH the source machine and D: after a
sync: if the final FINGERPRINT line matches, every tracked source file is
byte-identical. If it differs, the per-file table shows exactly which file is
stale or corrupted.

    python verify_sync.py

Only source files are hashed (data, checkpoints, images, venvs are ignored), so
the fingerprint depends solely on code — the same on any machine with the same
commit checked out.
"""
import hashlib
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {".git", "__pycache__", "venv", ".venv", "candy_env", "torch",
             "cropped_images", "cropped_masks", "cropped_noised_data",
             "noise_test_results", "sen1floods11", "imgs", "checkpoint",
             "raw_data", "raw_labels", ".idea", ".vscode", "runs", "_orphan_masks"}
EXTS = {".py"}


def iter_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS and not d.startswith("_archive_"))
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1] in EXTS:
                yield os.path.relpath(os.path.join(dirpath, fn), ROOT)


def main():
    combined = hashlib.sha256()
    rows = []
    for rel in sorted(iter_files()):
        with open(os.path.join(ROOT, rel), "rb") as f:
            data = f.read()
        h = hashlib.sha256(data).hexdigest()
        rows.append((rel, h, len(data)))
        combined.update(rel.replace(os.sep, "/").encode())
        combined.update(h.encode())

    w = max((len(r[0]) for r in rows), default=10)
    for rel, h, n in rows:
        print(f"{rel:<{w}}  {h[:16]}  {n:>7d} B")
    print("-" * (w + 30))
    print(f"{len(rows)} files")
    print(f"FINGERPRINT: {combined.hexdigest()[:32]}")


if __name__ == "__main__":
    main()
