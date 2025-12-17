import os
import shutil
from pathlib import Path
from collections import Counter
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A

DATASET_ROOT = "weapon_detection"
TRAIN_IMAGES = os.path.join(DATASET_ROOT, "train", "images")
TRAIN_LABELS = os.path.join(DATASET_ROOT, "train", "labels")
VAL_IMAGES = os.path.join(DATASET_ROOT, "val", "images")
VAL_LABELS = os.path.join(DATASET_ROOT, "val", "labels")

AUGMENTED_TRAIN_IMAGES = os.path.join(DATASET_ROOT, "train_augmented", "images")
AUGMENTED_TRAIN_LABELS = os.path.join(DATASET_ROOT, "train_augmented", "labels")

NUM_CLASSES = 9
CLASS_NAMES = ['Automatic Rifle', 'Bazooka', 'Grenade Launcher', 'Handgun',
               'Knife', 'Shotgun', 'SMG', 'Sniper', 'Sword']


def validate_images(image_dir, label_dir):
    corrupted_files = []
    image_files = list(Path(image_dir).glob("*.jpeg")) + list(Path(image_dir).glob("*.png"))

    for img_path in image_files:
        try:
            img = Image.open(img_path)
            img.verify()
        except Exception as e:
            corrupted_files.append(img_path)
            img_path.unlink()
            label_path = Path(label_dir) / f"{img_path.stem}.txt"
            if label_path.exists():
                label_path.unlink()
            print(f"{img_path.name}")

    if corrupted_files:
        print(f"{len(corrupted_files)}")
    return corrupted_files


def validate_labels(image_dir, label_dir):
    invalid_labels = []
    image_files = list(Path(image_dir).glob("*.jpeg")) + list(Path(image_dir).glob("*.png"))

    for img_path in image_files:
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"{img_path.name}")
            invalid_labels.append(img_path)
            img_path.unlink()

    return invalid_labels


def analyze_class_distribution(image_dir, dataset_name="Dataset"):
    class_counter = Counter()
    image_files = list(Path(image_dir).glob("*.jpeg")) + list(Path(image_dir).glob("*.png"))

    for img_path in image_files:
        filename = img_path.stem
        class_name = filename.split('_')[0]
        class_counter[class_name] += 1

    plot_bar_chart(class_counter, dataset_name)
    plot_pie_chart(class_counter, dataset_name)

    return class_counter


def plot_bar_chart(class_counter, dataset_name):
    plt.figure(figsize=(12, 6))
    classes = sorted(class_counter.keys())
    counts = [class_counter[c] for c in classes]

    plt.bar(classes, counts, color='#3498db', edgecolor='black', linewidth=1.5)
    plt.xlabel('Klasa', fontsize=12, fontweight='bold')
    plt.ylabel('Liczba obrazów', fontsize=12, fontweight='bold')
    plt.title(f'Rozkład klas - {dataset_name}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    output_filename = f'bar_chart_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Wykres słupkowy: {output_filename}")
    plt.close()


def plot_pie_chart(class_counter, dataset_name):
    plt.figure(figsize=(10, 8))
    classes = sorted(class_counter.keys())
    counts = [class_counter[c] for c in classes]
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))

    plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90,
            colors=colors, textprops={'fontsize': 10})
    plt.title(f'Rozkład klas - {dataset_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_filename = f'pie_chart_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()


def augment_yolo_dataset(image_dir, label_dir, output_image_dir, output_label_dir, num_augmentations=3):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    transform = A.Compose(
        [
            A.Rotate(limit=30, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )

    image_files = list(Path(image_dir).glob("*.jpg")) \
                 + list(Path(image_dir).glob("*.jpeg")) \
                 + list(Path(image_dir).glob("*.png"))

    augmented_counter = Counter()
    sample_images = []

    for img_path in image_files:
        shutil.copy(img_path, os.path.join(output_image_dir, img_path.name))
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy(label_path, os.path.join(output_label_dir, label_path.name))

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        bboxes = []
        class_labels = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    class_labels.append(class_id)
                    bboxes.append([x, y, w, h])

        if len(bboxes) == 0:
            continue

        class_name = img_path.stem.split("_")[0]

        for aug_idx in range(num_augmentations):
            try:
                augmented = transform(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels,
                )

                if len(augmented["bboxes"]) == 0:
                    continue

                aug_img_name = f"{img_path.stem}_aug_{aug_idx}{img_path.suffix}"
                aug_img_path = os.path.join(output_image_dir, aug_img_name)
                cv2.imwrite(
                    aug_img_path,
                    cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR),
                )

                aug_label_path = os.path.join(
                    output_label_dir, f"{img_path.stem}_aug_{aug_idx}.txt"
                )
                with open(aug_label_path, "w") as f:
                    for bbox, class_id in zip(augmented["bboxes"], augmented["class_labels"]):
                        x_center, y_center, width, height = bbox
                        f.write(
                            f"{int(class_id)} "
                            f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        )

                augmented_counter[class_name] += 1

            except Exception as e:
                print(f"{img_path.name}: {e}")

    return augmented_counter, sample_images


def main():
    validate_images(TRAIN_IMAGES, TRAIN_LABELS)
    validate_labels(TRAIN_IMAGES, TRAIN_LABELS)

    analyze_class_distribution(TRAIN_IMAGES, "Train Set (Przed augmentacją)")

    augment_yolo_dataset(
        image_dir=TRAIN_IMAGES,
        label_dir=TRAIN_LABELS,
        output_image_dir=AUGMENTED_TRAIN_IMAGES,
        output_label_dir=AUGMENTED_TRAIN_LABELS,
        num_augmentations=3
    )

    analyze_class_distribution(AUGMENTED_TRAIN_IMAGES, "Train Set (Po augmentacji)")


if __name__ == "__main__":
    main()
