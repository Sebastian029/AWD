from ultralytics import YOLO
import pandas as pd
import json

# Lista eksperymentów do przetestowania
experiments = [
    {"model": "yolov8n.pt", "epochs": 20, "imgsz": 640, "batch": 32},
    {"model": "yolov8s.pt", "epochs": 20, "imgsz": 640, "batch": 32},
    {"model": "yolov8n.pt", "epochs": 20, "imgsz": 800, "batch": 16},
    {"model": "yolov8n.pt", "epochs": 20, "imgsz": 640, "batch": 64},
]


results_data = []

for idx, exp in enumerate(experiments):
    print(f"\n=== Experiment {idx + 1}/{len(experiments)} ===")
    print(f"Config: {exp}")

    model = YOLO(exp["model"])

    results = model.train(
        data="data.yaml",
        epochs=exp["epochs"],
        imgsz=exp["imgsz"],
        batch=exp["batch"],
        name=f"exp_{idx}",
        save=True,
        plots=True
    )

    metrics = model.val()

    experiment_results = {
        "experiment": idx,
        "model_name": exp["model"],
        "epochs": exp["epochs"],
        "imgsz": exp["imgsz"],
        "batch": exp["batch"],
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "precision": metrics.box.p.mean(),
        "recall": metrics.box.r.mean(),
        "f1": 2 * (metrics.box.p.mean() * metrics.box.r.mean()) / (metrics.box.p.mean() + metrics.box.r.mean()),
    }

    results_data.append(experiment_results)

    #model.save(f"weapon_detection_exp_{idx}.pt")

df = pd.DataFrame(results_data)
df.to_csv("yolo_experiments_comparison.csv", index=False)
print("\n=== Summary ===")
print(df)

best_model = df.loc[df['mAP50-95'].idxmax()]
print(f"\nBest model: Experiment {best_model['experiment']}")
print(best_model)
