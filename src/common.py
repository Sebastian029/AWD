"""Shared task registry and helpers. All scripts are meant to be run from the repo root."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)  # configs use paths relative to the repo root, so scripts work from any directory

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# "weapons" is the main task of this repo; "masks" is the second half of the shared (two-person) project
# and was done by the project partner.
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
        "kaggle": "https://www.kaggle.com/datasets/snehilsanyal/weapon-detection-test",
        "layout": "data/weapons/{train,val}/{images,labels}",
    },
    "masks": {  # shared project part (partner)
        "root": Path("data/masks"),
        "train_images": "images/train",
        "train_labels": "labels/train",
        "val_labels": "labels/valid",
        "aug_dir": "train_augmented",
        "names": ["No Mask", "Mask"],
        "metadata": None,
        "label": "Masks",
        "kaggle": "https://www.kaggle.com/datasets/aditya276/face-mask-dataset-yolo-format",
        "layout": "data/masks/{images,labels}/{train,valid,test}",
    },
}


def get_task(name):
    return TASKS[name]


def config_path(task, variant="data"):
    """variant: data (YOLO built-in aug) | data_raw (no aug) | data_aug (offline Albumentations)."""
    return Path("configs") / task / f"{variant}.yaml"


def list_images(directory):
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def require_data(task):
    """Exit with a pointer to the Kaggle dataset when the data folder is missing."""
    t = TASKS[task]
    if not (t["root"] / t["train_images"]).is_dir():
        sys.exit(f"Dataset for '{task}' not found.\n"
                 f"Download it from {t['kaggle']}\n"
                 f"and unpack it as {t['layout']}")
