from ultralytics import YOLO

# Załaduj model (np. YOLOv8n - najmniejszy dostępny model do szybkiego trenowania)
model = YOLO("yolov8n.pt")

# Trening modelu na Twoim datasetcie
# data.yaml - plik konfiguracyjny datasetu
# epochs - liczba epok trenowania
results = model.train(data="data.yaml", epochs=10, imgsz=640)

# Testowanie na zbiorze walidacyjnym (automatycznie zdefiniowanym w data.yaml)
metrics = model.val()

# Detekcja na pojedynczym obrazie (ścieżka lub URL)
results = model("weapon_detection/val/images/Automatic Rifle_9.jpeg")

# Wyświetlanie wyników detekcji (np. bounding box + label)
for result in results:
    result.show()

# Zapis modelu wytrenowanego do pliku (opcjonalnie)
model.save("yolo_weapon_detection_trained.pt")
