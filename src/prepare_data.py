"""Validate the training split, plot class distribution and build the offline-augmented train set.

    python src/prepare_data.py --task weapons
    python src/prepare_data.py --task masks --delete-invalid
"""
import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import get_task, list_images

PLOTS_DIR = Path("results/plots")
NUM_AUGMENTATIONS = 3
SEED = 42


def validate(image_dir, label_dir, delete=False):
    """Report (and optionally delete) unreadable images and images without a label file."""
    bad = []
    for img_path in list_images(image_dir):
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        try:
            with Image.open(img_path) as img:
                img.verify()
            ok, reason = label_path.exists(), "missing label"
        except Exception:
            ok, reason = False, "corrupted image"
        if not ok:
            bad.append((img_path, reason))
            print(f"  [{reason}] {img_path.name}")
            if delete:
                img_path.unlink()
                label_path.unlink(missing_ok=True)
    print(f"Validation: {len(bad)} invalid file(s)" + (" (deleted)" if delete and bad else ""))
    return bad


def class_distribution(label_dir, names, title, out_name):
    counter = Counter()
    for label_path in Path(label_dir).glob("*.txt"):
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                cid = int(parts[0])
                counter[names[cid] if 0 <= cid < len(names) else f"class_{cid}"] += 1

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    classes = [n for n in names if n in counter]
    plt.figure(figsize=(12, 6))
    plt.bar(classes, [counter[c] for c in classes], color="#3498db", edgecolor="black")
    plt.xlabel("Class")
    plt.ylabel("Number of boxes")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / out_name, dpi=200)
    plt.close()
    return counter


def read_yolo_labels(label_path):
    bboxes, class_ids = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                class_ids.append(int(parts[0]))
                bboxes.append([float(v) for v in parts[1:]])
    return bboxes, class_ids


def augment_dataset(image_dir, label_dir, out_image_dir, out_label_dir, n_aug=NUM_AUGMENTATIONS):
    """Copy the originals and add n_aug Albumentations variants of each image."""
    random.seed(SEED)
    np.random.seed(SEED)
    Path(out_image_dir).mkdir(parents=True, exist_ok=True)
    Path(out_label_dir).mkdir(parents=True, exist_ok=True)

    transform = A.Compose(
        [
            A.Rotate(limit=30, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )

    created = 0
    for img_path in list_images(image_dir):
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        shutil.copy(img_path, Path(out_image_dir) / img_path.name)
        if label_path.exists():
            shutil.copy(label_path, Path(out_label_dir) / label_path.name)

        bboxes, class_ids = read_yolo_labels(label_path)
        if not bboxes:
            continue
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        for i in range(n_aug):
            try:
                out = transform(image=image, bboxes=bboxes, class_labels=class_ids)
            except Exception as e:
                print(f"  {img_path.name}: {e}")
                continue
            if not out["bboxes"]:
                continue
            stem = f"{img_path.stem}_aug_{i}"
            cv2.imwrite(str(Path(out_image_dir) / f"{stem}{img_path.suffix}"),
                        cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR))
            with open(Path(out_label_dir) / f"{stem}.txt", "w") as f:
                for (x, y, w, h), cid in zip(out["bboxes"], out["class_labels"]):
                    f.write(f"{int(cid)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            created += 1
    print(f"Augmentation: {created} new image(s) written to {out_image_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["weapons", "masks"], required=True)
    ap.add_argument("--delete-invalid", action="store_true",
                    help="delete corrupted images / images without labels (default: only report)")
    args = ap.parse_args()

    t = get_task(args.task)
    root = t["root"]
    train_images, train_labels = root / t["train_images"], root / t["train_labels"]
    aug_images, aug_labels = root / t["aug_dir"] / "images", root / t["aug_dir"] / "labels"

    validate(train_images, train_labels, delete=args.delete_invalid)
    class_distribution(train_labels, t["names"], f"{t['label']}: train set (before augmentation)",
                       f"dist_{args.task}_before_aug.png")
    augment_dataset(train_images, train_labels, aug_images, aug_labels)
    class_distribution(aug_labels, t["names"], f"{t['label']}: train set (after augmentation)",
                       f"dist_{args.task}_after_aug.png")


if __name__ == "__main__":
    main()
