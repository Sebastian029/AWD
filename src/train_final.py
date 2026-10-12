"""Train the best configuration found by the grid search and evaluate it with full plots.

    python src/train_final.py --task weapons
"""
import argparse
import time

from ultralytics import YOLO

from common import config_path, require_data

# Winner of the grid search for both tasks.
MODEL, EPOCHS, BATCH, OPTIMIZER, LR0, PATIENCE = "yolo11m.pt", 200, 16, "SGD", 0.001, 10


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["weapons", "masks"], required=True)
    args = ap.parse_args()

    require_data(args.task)
    data = str(config_path(args.task, "data"))
    project = f"runs/{args.task}"

    model = YOLO(MODEL)
    start = time.time()
    model.train(data=data, epochs=EPOCHS, batch=BATCH, optimizer=OPTIMIZER, lr0=LR0,
                patience=PATIENCE, plots=True, project=project, name="final_yolo11m", exist_ok=True)
    minutes = (time.time() - start) / 60

    # masks has a held-out test split; weapons only has val (cleaned of train duplicates, see make_clean_val.py)
    split = "test" if args.task == "masks" else "val"
    m = model.val(data=data, split=split, save=True, plots=True, project=project,
                  name="final_eval", exist_ok=True)
    print(f"\n===== FINAL METRICS ({split} split) =====")
    print(f"Model:      {MODEL}")
    print(f"mAP@50:     {m.box.map50:.4f}")
    print(f"mAP@50-95:  {m.box.map:.4f}")
    print(f"Precision:  {m.box.mp:.4f}")
    print(f"Recall:     {m.box.mr:.4f}")
    print(f"Train time: {minutes:.1f} min")


if __name__ == "__main__":
    main()
