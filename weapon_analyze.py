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


def compare_dataset_sizes(original_dir, augmented_dir):
    """
    Porównuje liczbę obrazów przed i po augmentacji.
    """
    # Policz pliki
    original_count = len(list(Path(original_dir).glob("*.jpeg")))
    augmented_count = len(list(Path(augmented_dir).glob("*.jpeg")))

    # Przygotuj dane do wykresu
    categories = ['Przed augmentacją', 'Po augmentacji']
    counts = [original_count, augmented_count]
    colors = ['#3498db', '#2ecc71']

    # Stwórz wykres
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, counts, color=colors, width=0.6, edgecolor='black', linewidth=1.5)

    # Dodaj wartości na słupkach
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Oblicz i wyświetl wzrost
    increase = augmented_count - original_count
    increase_percent = (increase / original_count) * 100

    plt.ylabel('Liczba obrazów', fontsize=12, fontweight='bold')
    plt.title(f'Porównanie wielkości datasetu\nWzrost: +{increase} obrazów (+{increase_percent:.1f}%)',
              fontsize=14, fontweight='bold')
    plt.ylim(0, augmented_count * 1.15)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Zapisz wykres
    output_filename = 'dataset_size_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Wykres porównania zapisany jako: {output_filename}")
    print(f"✓ Przed augmentacją: {original_count} obrazów")
    print(f"✓ Po augmentacji: {augmented_count} obrazów")
    print(f"✓ Wzrost: +{increase} obrazów (+{increase_percent:.1f}%)")

    plt.close()


def augment_yolo_dataset(image_dir, label_dir, output_image_dir, output_label_dir, num_augmentations=3):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # Prosty pipeline - tylko podstawowe augmentacje
    transform = A.Compose([
        # OBRÓT
        A.Rotate(limit=30, p=0.5),

        # SKALOWANIE I PRZESUNIĘCIE
        A.ShiftScaleRotate(
            shift_limit=0.1,  # Przesunięcie ±10%
            scale_limit=0.2,  # Skalowanie ±20%
            rotate_limit=0,  # Obrót wyłączony (mamy osobno)
            p=0.5
        ),

        # ZMIANA KOLORU
        A.ColorJitter(
            brightness=0.2,  # Jasność
            contrast=0.2,  # Kontrast
            saturation=0.2,  # Nasycenie
            hue=0.1,  # Odcień
            p=0.5
        ),

        # SZUM GAUSSOWSKI
        A.GaussNoise(
            var_limit=(10.0, 50.0),
            p=0.3
        ),

    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

    # Pobierz listę obrazów
    image_files = list(Path(image_dir).glob("*.jpeg"))
    augmented_count = 0

    print(f"\nPrzetwarzanie {len(image_files)} obrazów...")

    for img_path in tqdm(image_files, desc="Augmentacja"):
        shutil.copy(img_path, os.path.join(output_image_dir, img_path.name))
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy(label_path, os.path.join(output_label_dir, label_path.name))

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        bboxes, class_labels = [], []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_labels.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:]])

        if len(bboxes) == 0:
            continue

        for aug_idx in range(num_augmentations):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

                if len(augmented['bboxes']) == 0:
                    continue

                aug_img_name = f"{img_path.stem}_aug_{aug_idx}{img_path.suffix}"
                aug_img_path = os.path.join(output_image_dir, aug_img_name)
                cv2.imwrite(aug_img_path, cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))

                aug_label_path = os.path.join(output_label_dir, f"{img_path.stem}_aug_{aug_idx}.txt")
                with open(aug_label_path, 'w') as f:
                    for bbox, class_id in zip(augmented['bboxes'], augmented['class_labels']):
                        x_center, y_center, width, height = bbox
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

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

    print("\n[KROK 4/6] Walidacja danych walidacyjnych...")
    validate_images(VAL_IMAGES, VAL_LABELS)
    validate_labels(VAL_IMAGES, VAL_LABELS)
    val_distribution = analyze_class_distribution(VAL_IMAGES, "Validation Set")

    print("\n[KROK 5/6] Augmentacja danych treningowych...")
    augment_yolo_dataset(
        image_dir=TRAIN_IMAGES,
        label_dir=TRAIN_LABELS,
        output_image_dir=AUGMENTED_TRAIN_IMAGES,
        output_label_dir=AUGMENTED_TRAIN_LABELS,
        num_augmentations=3
    )

    print("\n[KROK 6/6] Porównanie wielkości datasetu przed i po augmentacji...")
    compare_dataset_sizes(TRAIN_IMAGES, AUGMENTED_TRAIN_IMAGES)


if __name__ == "__main__":
    main()
