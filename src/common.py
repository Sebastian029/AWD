"""Shared task registry and helpers. All scripts are meant to be run from the repo root."""
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

TASKS = {
    "weapons": {
        "root": Path("data/weapons"),
        "train_images": "train/images",
        "train_labels": "train/labels",
        "val_labels": "val/labels",
        "aug_dir": "train_augmented",
        "names": ["Automatic Rifle", "Bazooka", "Grenade Launcher", "Handgun",
                  "Knife", "Shotgun", "SMG", "Sniper", "Sword"],
        "metadata": Path("metadata/weapons_metadata.csv"),
        "label": "Weapons",
    },
    "masks": {
        "root": Path("data/masks"),
        "train_images": "images/train",
        "train_labels": "labels/train",
        "val_labels": "labels/valid",
        "aug_dir": "train_augmented",
        "names": ["No Mask", "Mask"],
        "metadata": None,
        "label": "Masks",
    },
}


def get_task(name):
    return TASKS[name]


def config_path(task, variant="data"):
    """variant: data (YOLO built-in aug) | data_raw (no aug) | data_aug (offline Albumentations)."""
    return Path("configs") / task / f"{variant}.yaml"


def list_images(directory):
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTS)
