"""Dataset sanity checks: split sizes, label/filename consistency and near-duplicates across splits.

    python src/check_dataset.py --task weapons
    python src/check_dataset.py --task masks --max-dist 6
"""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from common import get_task, list_images, require_data


def find_splits(root):
    """Return {split: (image_dir, label_dir)} for both supported layouts."""
    splits = {}
    for name in ("train", "val", "valid", "test"):
        if (root / name / "images").is_dir():                       # weapons: split/images
            splits[name] = (root / name / "images", root / name / "labels")
        elif (root / "images" / name).is_dir():                     # masks: images/split
            splits[name] = (root / "images" / name, root / "labels" / name)
    return splits


def dhash(path, size=8):
    """64-bit difference hash: robust to resizing and mild re-compression."""
    with Image.open(path) as img:
        g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def check_filename_vs_label(names, image_dir, label_dir, split):
    """Weapons files are named '<Class>_<n>.jpeg'; compare that class with the id in the label."""
    mismatches = 0
    for img in list_images(image_dir):
        label = Path(label_dir) / f"{img.stem}.txt"
        if not label.exists():
            continue
        ids = {int(l.split()[0]) for l in label.read_text().splitlines() if l.split()}
        expected = img.stem.rsplit("_", 1)[0]
        if expected in names and ids != {names.index(expected)}:
            mismatches += 1
            print(f"  [{split}] {img.name}: label ids {sorted(ids)} vs name -> {names.index(expected)}")
    print(f"[{split}] filename/label class mismatches: {mismatches}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["weapons", "masks"], default="weapons")
    ap.add_argument("--max-dist", type=int, default=4,
                    help="max Hamming distance (of 64 bits) to call two images near-duplicates")
    args = ap.parse_args()

    require_data(args.task)
    t = get_task(args.task)
    splits = find_splits(t["root"])

    print("== Split sizes ==")
    hashes = {}
    for split, (img_dir, lbl_dir) in splits.items():
        images = list_images(img_dir)
        labels = list(Path(lbl_dir).glob("*.txt")) if Path(lbl_dir).is_dir() else []
        classes = Counter()
        for lp in labels:
            for line in lp.read_text().splitlines():
                if line.split():
                    classes[int(line.split()[0])] += 1
        print(f"{split:6s} images={len(images):4d} labels={len(labels):4d} boxes per class={dict(sorted(classes.items()))}")
        hashes[split] = [(p, dhash(p)) for p in images]

    if args.task == "weapons":
        print("\n== Filename vs label class ==")
        for split, (img_dir, lbl_dir) in splits.items():
            check_filename_vs_label(t["names"], img_dir, lbl_dir, split)

    print(f"\n== Near-duplicates between splits (Hamming <= {args.max_dist}) ==")
    total = 0
    for a, b in combinations(hashes, 2):
        pairs = []
        for pa, ha in hashes[a]:
            for pb, hb in hashes[b]:
                d = int(np.count_nonzero(ha != hb))
                if d <= args.max_dist:
                    pairs.append((d, pa.name, pb.name))
        total += len(pairs)
        print(f"{a} vs {b}: {len(pairs)} pair(s)")
        for d, na, nb in sorted(pairs)[:10]:
            print(f"    d={d}  {na}  <->  {nb}")
    print(f"\nTotal cross-split near-duplicate pairs: {total}")


if __name__ == "__main__":
    main()
