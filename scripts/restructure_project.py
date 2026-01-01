"""Helper script to migrate project to src/ layout.

WARNING: This script attempts to move files and update imports. Run it in a branch and review changes.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PKG = SRC / "seq2seq_sar"

FILES_TO_MOVE = ["seq2seq_sar"]


def ensure_dirs():
    SRC.mkdir(parents=True, exist_ok=True)
    PKG.mkdir(parents=True, exist_ok=True)


def copy_package():
    for name in FILES_TO_MOVE:
        src_path = ROOT / name
        if src_path.exists():
            dst = PKG / src_path.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src_path, dst)
            print(f"Copied {src_path} -> {dst}")
        else:
            print(f"{src_path} not found; skipping")


if __name__ == "__main__":
    ensure_dirs()
    copy_package()
    print("Done. Review files and adjust imports/tests as needed.")
