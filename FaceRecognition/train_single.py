import time
from ultralytics import YOLO


def main():
    # --- KONFIGURACJA ZWYCIĘSKA ---
    DATA_YAML = "face_data.yaml"
    MODEL_WEIGHTS = "yolo11m.pt"
    EPOCHS = 200
    BATCH = 16
    OPTIMIZER = "SGD"
    LR0 = 0.001

    # --- ŁADOWANIE MODELU ---
    print(f"🔄 Ładowanie modelu: {MODEL_WEIGHTS}...")
    model = YOLO(MODEL_WEIGHTS)

    print(f"🚀 Start treningu (Opt: {OPTIMIZER}, LR: {LR0})...")
    start_time = time.time()

    # --- TRENING ---
    # Tutaj plots=True jest domyślne, ale warto zostawić dla pewności.
    # To wygeneruje: results.png, confusion_matrix, train_batch*.jpg
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        optimizer=OPTIMIZER,
        lr0=LR0,
        patience=10,
        plots=True,  # KLUCZOWE: Generuje wykresy uczenia
        verbose=True,
        project="final_training",
        name="yolo11m_weapons_best"
    )

    end_time = time.time()
    train_duration = end_time - start_time
    print(f"⏱ Czas treningu: {train_duration / 60:.1f} min")

    # --- WALIDACJA I METRYKI ---
    print("\n📊 Walidacja modelu na zbiorze testowym...")

    # Tutaj generujemy "dowody" działania modelu na czystym zbiorze testowym
    metrics = model.val(
        data=DATA_YAML,
        split="val",
        save=True,  # KLUCZOWE: Zapisuje zdjęcia JPG z wykrytą bronią
        plots=True,  # Generuje macierz pomyłek dla zbioru testowego
        project="final_training",
        name="test_evaluation"  # Wyniki będą w: final_training/test_evaluation
    )

    # GŁÓWNE METRYKI DETEKCJI
    map50 = metrics.box.map50
    map5095 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr
    fitness = metrics.fitness

    print("\n===== METRYKI KOŃCOWE =====")
    print(f"Model:        {MODEL_WEIGHTS}")
    print(f"mAP@50:       {map50:.4f}")
    print(f"mAP@50-95:    {map5095:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"Fitness:      {fitness:.4f}")
    print(f"Czas treningu: {train_duration / 60:.1f} min")
    print("===========================")


if __name__ == "__main__":
    main()
