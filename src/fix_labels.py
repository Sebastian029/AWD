"""Overwrite the class id in YOLO label files with the `target` from the Kaggle metadata.csv.

    python src/fix_labels.py --split val
"""
import argparse
import csv
from pathlib import Path

from common import get_task


def load_mapping(metadata_path):
    with open(metadata_path, newline="", encoding="utf-8-sig") as f:
        return {r["labelfile"].strip(): r["target"].strip() for r in csv.DictReader(f)}


def fix_labels(label_dir, mapping):
    fixed = skipped = 0
    for path in Path(label_dir).glob("*.txt"):
        if path.name == "classes.txt":
            continue
        target = mapping.get(path.name)
        if target is None:
            skipped += 1
            continue
        lines, changed = [], False
        for line in path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] != target:
                parts[0], changed = target, True
            lines.append(" ".join(parts))
        if changed:
            path.write_text("\n".join(lines) + "\n")
            fixed += 1
    print(f"Fixed {fixed} file(s); {skipped} not in metadata")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    args = ap.parse_args()

    t = get_task("weapons")
    label_dir = t["root"] / (t["train_labels"] if args.split == "train" else t["val_labels"])
    fix_labels(label_dir, load_mapping(t["metadata"]))


if __name__ == "__main__":
    main()
