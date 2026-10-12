"""Build data/weapons/val_clean: the val images that have no near-duplicate in the train split.

The Kaggle weapons set contains copies of the same picture in train and val, which inflates val metrics.
Originals are left untouched; clean copies are written next to them.

    python src/make_clean_val.py
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

from check_dataset import dhash
from common import get_task, list_images, require_data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-dist", type=int, default=4)
    args = ap.parse_args()

    require_data("weapons")
    root = get_task("weapons")["root"]
    train = [dhash(p) for p in list_images(root / "train/images")]
    out_img, out_lbl = root / "val_clean/images", root / "val_clean/labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    kept = dropped = 0
    for img in list_images(root / "val/images"):
        h = dhash(img)
        if any(np.count_nonzero(h != t) <= args.max_dist for t in train):
            dropped += 1
            continue
        shutil.copy(img, out_img / img.name)
        label = root / "val/labels" / f"{img.stem}.txt"
        if label.exists():
            shutil.copy(label, out_lbl / label.name)
        kept += 1
    print(f"val_clean: kept {kept}, dropped {dropped} duplicate(s) of train images")


if __name__ == "__main__":
    main()
