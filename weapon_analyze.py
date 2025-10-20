import os
import shutil
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm


DATASET_ROOT = "weapon_detection"
TRAIN_IMAGES = os.path.join(DATASET_ROOT, "train", "images")
TRAIN_LABELS = os.path.join(DATASET_ROOT, "train", "labels")
VAL_IMAGES = os.path.join(DATASET_ROOT, "val", "images")
VAL_LABELS = os.path.join(DATASET_ROOT, "val", "labels")

AUGMENTED_TRAIN_IMAGES = os.path.join(DATASET_ROOT, "train_augmented", "images")
AUGMENTED_TRAIN_LABELS = os.path.join(DATASET_ROOT, "train_augmented", "labels")

NUM_CLASSES = 9


def validate_images(image_dir, label_dir):
    corrupted_files = []
    image_files = Path(image_dir).glob("*.jpeg")

    for img_path in image_files:
        try:
            img = Image.open(img_path)
            img.verify()
            img = Image.open(img_path)
            img.load()

        except Exception as e:
            corrupted_files.append(img_path)

            img_path.unlink()
            label_path = Path(label_dir) / f"{img_path.stem}.txt"
            if label_path.exists():
                label_path.unlink()

    return corrupted_files


def validate_labels(image_dir, label_dir):
    invalid_labels = []
    image_files = Path(image_dir).glob("*.jpeg")

    for img_path in image_files:
        label_path = Path(label_dir) / f"{img_path.stem}.txt"

        if not label_path.exists():
            print(f"\nBrak etykiety dla: {img_path.name}")
            invalid_labels.append(img_path)
            img_path.unlink()
            continue



def analyze_class_distribution(image_dir, dataset_name="Dataset"):
    class_counter = Counter()

    image_files = Path(image_dir).glob("*.jpeg")
    total_images = 0

    for img_path in image_files:
        filename = img_path.stem
        class_name = filename.split('_')[0]
        class_counter[class_name] += 1
        total_images += 1


    plt.figure(figsize=(12, 6))
    classes = list(class_counter.keys())
    counts = list(class_counter.values())

    plt.bar(classes, counts)
    plt.xlabel('Klasa', fontsize=12)
    plt.ylabel('Liczba obrazów', fontsize=12)
    plt.title(f'Rozkład klas w zbiorze {dataset_name}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_filename = f'{dataset_name.lower().replace(" ", "_")}_class_distribution.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\n✓ Wykres zapisany jako: {output_filename}")

    return class_counter


def create_augmentation_pipeline():
    transform = A.Compose([
        A.OneOf([
            A.Rotate(limit=15, p=1.0),  # Obrót ±15 stopni
            A.Rotate(limit=30, p=1.0),  # Obrót ±30 stopni
        ], p=0.5),

        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=0, p=1.0),  # Przesunięcie i skalowanie
            A.HorizontalFlip(p=1.0),  # Odbicie poziome
        ], p=0.5),

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),  # Jasność/kontrast
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),  # Kolor
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),  # Przesunięcie RGB
        ], p=0.5),

        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),  # Szum gaussowski
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),  # Rozmycie gaussowskie
        ], p=0.3),

    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

    return transform


def read_yolo_labels(label_path):
    bboxes = []
    class_labels = []

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    #bbox = list(map(float, parts[1:]))
                    bbox = [float(x) for x in parts[1:]]
                    class_labels.append(class_id)
                    bboxes.append(bbox)

    return bboxes, class_labels


def save_yolo_labels(label_path, bboxes, class_labels):
    with open(label_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            x_center, y_center, width, height = bbox
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def augment_dataset(image_dir, label_dir, output_image_dir, output_label_dir, num_augmentations=3):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Skopiuj oryginalne dane
    print("\nKopiowanie oryginalnych danych...")
    image_files = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))

    for img_path in tqdm(image_files, desc="Kopiowanie"):
        # Kopiuj obraz
        shutil.copy(img_path, os.path.join(output_image_dir, img_path.name))

        # Kopiuj etykietę
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy(label_path, os.path.join(output_label_dir, label_path.name))

    # Pipeline augmentacji
    transform = create_augmentation_pipeline()

    print(f"\nGenerowanie augmentowanych danych...")
    augmented_count = 0

    for img_path in tqdm(image_files, desc="Augmentacja"):
        # Wczytaj obraz
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Wczytaj etykiety
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        bboxes, class_labels = read_yolo_labels(label_path)

        if len(bboxes) == 0:
            continue

        # Wykonaj num_augmentations augmentacji
        for aug_idx in range(num_augmentations):
            try:
                # Zastosuj augmentację
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_class_labels = augmented['class_labels']

                if len(aug_bboxes) == 0:
                    continue

                # Zapisz augmentowany obraz
                aug_img_name = f"{img_path.stem}_aug_{aug_idx}{img_path.suffix}"
                aug_img_path = os.path.join(output_image_dir, aug_img_name)
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(aug_img_path, aug_image_bgr)

                # Zapisz augmentowane etykiety
                aug_label_name = f"{img_path.stem}_aug_{aug_idx}.txt"
                aug_label_path = os.path.join(output_label_dir, aug_label_name)
                save_yolo_labels(aug_label_path, aug_bboxes, aug_class_labels)

                augmented_count += 1

            except Exception as e:
                print(f"\nBłąd podczas augmentacji {img_path.name}: {e}")
                continue

    print(f"\n✓ Wygenerowano {augmented_count} augmentowanych obrazów")
    print(f"✓ Łącznie obrazów: {len(list(Path(output_image_dir).glob('*')))}")


# ==========================
# MAIN
# ==========================

def main():
    print("\n[KROK 1/6] Walidacja obrazów treningowych...")
    validate_images(TRAIN_IMAGES, TRAIN_LABELS)

    print("\n[KROK 2/6] Walidacja etykiet treningowych...")
    validate_labels(TRAIN_IMAGES, TRAIN_LABELS)

    print("\n[KROK 3/6] Analiza rozkładu klas treningowych...")
    train_distribution = analyze_class_distribution(TRAIN_IMAGES, "Train Set")

    # WSTĘPNA OBRÓBKA - VAL
    print("\n[KROK 4/6] Walidacja danych walidacyjnych...")
    validate_images(VAL_IMAGES, VAL_LABELS)
    validate_labels(VAL_IMAGES, VAL_LABELS)
    val_distribution = analyze_class_distribution(VAL_IMAGES, "Validation Set")

    # 2. AUGMENTACJA
    print("\n[KROK 5/6] Augmentacja danych treningowych...")
    augment_dataset(
        image_dir=TRAIN_IMAGES,
        label_dir=TRAIN_LABELS,
        output_image_dir=AUGMENTED_TRAIN_IMAGES,
        output_label_dir=AUGMENTED_TRAIN_LABELS,
        num_augmentations=3  # Każdy obraz x3 augmentacje
    )






if __name__ == "__main__":
    main()
