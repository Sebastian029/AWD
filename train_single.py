import time
from ultralytics import YOLO


def main():
    # --- KONFIGURACJA USTAWIEŃ ---
    DATA_YAML = "data.yaml"        # ścieżka do Twojego pliku z datasetem
    MODEL_WEIGHTS = "yolov8m.pt"   # start z pretrained nano
    EPOCHS = 100                     # rozsądne minimum
    IMG_SIZE = 640
    BATCH = 32
    OPTIMIZER = "SGD"              # stabilna, domyślna opcja
    LR0 = 0.001                     # standardowy lr dla SGD

    # --- ŁADOWANIE MODELU ---
    model = YOLO(MODEL_WEIGHTS)

    print("🚀 Start treningu...")
    start = time.time()

    # --- TRENING ---
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,  # dodaj
        optimizer=OPTIMIZER,  # dodaj
        lr0=LR0,  # dodaj
        patience=10,  # Early stopping po 10 epok bez poprawy
        plots=True,  # Krzywe uczenia
        verbose=True,
    )

    train_time = time.time() - start
    print(f"⏱ Czas treningu: {train_time / 60:.1f} min")

    # --- WALIDACJA I METRYKI ---
    print("\n📊 Walidacja modelu...")
    metrics = model.val(
        data=DATA_YAML,
        split="val",
        save=False,
        plots=True,
    )

    # GŁÓWNE METRYKI DETEKCJI
    map50 = float(metrics.box.map50)   # mAP@0.5
    map5095 = float(metrics.box.map)   # mAP@0.5:0.95
    precision = float(metrics.box.p.mean())
    recall = float(metrics.box.r.mean())
    fitness = float(metrics.fitness)

    print("\n===== METRYKI KOŃCOWE =====")
    print(f"mAP@50:       {map50:.4f}")
    print(f"mAP@50-95:    {map5095:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"Fitness:      {fitness:.4f}")
    print(f"Czas treningu: {train_time / 60:.1f} min")
    print("===========================")


if __name__ == "__main__":
    main()
