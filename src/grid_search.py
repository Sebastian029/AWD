"""Grid search over data variant x model x optimizer x lr0; appends one row per run to a CSV.

    python src/grid_search.py --task weapons
"""
import argparse
import itertools
import shutil
import time
import traceback
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from common import config_path, require_data

GRID = {
    "variant": ["data", "data_aug", "data_raw"],
    "model": ["yolo11m.pt", "yolo11n.pt"],
    "optimizer": ["SGD", "AdamW"],
    "lr0": [0.01, 0.001],
}
EPOCHS, BATCH, IMGSZ, PATIENCE = 100, 16, 640, 10

# YOLO's own augmentation is switched off for the offline (Albumentations) set and the raw baseline.
NO_YOLO_AUG = {
    "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0, "perspective": 0.0,
    "flipud": 0.0, "fliplr": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
    "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "erasing": 0.0, "auto_augment": None,
}
AUG_MODES = {
    "data": ("YOLO Internal", {}),
    "data_aug": ("Custom (Albumentations)", NO_YOLO_AUG),
    "data_raw": ("None (Raw Baseline)", NO_YOLO_AUG),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["weapons", "masks"], default="weapons")
    ap.add_argument("--seed", type=int, default=0, help="training seed; repeat with different seeds to estimate noise")
    ap.add_argument("--out", help="results CSV (default: results/<task>_grid.csv)")
    args = ap.parse_args()

    require_data(args.task)
    csv_path = Path(args.out or f"results/{args.task}_grid.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    runs_dir = Path("temp_runs")

    for exp_id, combo in enumerate(itertools.product(*GRID.values()), start=1):
        cfg = dict(zip(GRID.keys(), combo))
        data = str(config_path(args.task, cfg["variant"]))
        aug_label, aug_params = AUG_MODES[cfg["variant"]]
        val_split = "val_clean" if args.task == "weapons" else "valid"
        run_name = f"{args.task}_exp_{exp_id}"
        print(f"\n[{exp_id}] {data} | {cfg['model']} | {cfg['optimizer']} | lr0={cfg['lr0']} | aug={aug_label}")

        model = YOLO(cfg["model"])
        start = time.time()
        try:
            model.train(
                data=data, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                lr0=cfg["lr0"], optimizer=cfg["optimizer"], patience=PATIENCE,
                verbose=False, plots=False, seed=args.seed, name=run_name, project=str(runs_dir), exist_ok=True,
                **aug_params,
            )
            train_time = time.time() - start
            stopped_epoch = getattr(model.trainer, "epoch", -1)
            best_epoch = getattr(getattr(model.trainer, "stopper", None), "best_epoch", -1)

            m = model.val(data=data, split="val", save=False, plots=False)
            row = {
                "exp_id": exp_id, "data": data, "model": cfg["model"],
                "seed": args.seed, "best_epoch": best_epoch, "stopped_epoch": stopped_epoch, "optimizer": cfg["optimizer"], "lr0": cfg["lr0"],
                "augmentation_type": aug_label,
                "val_mAP50": float(m.box.map50), "val_mAP50-95": float(m.box.map),
                "val_precision": float(m.box.mp), "val_recall": float(m.box.mr),
                "fitness": float(m.fitness), "time_min": train_time / 60, "val_split": val_split,
            }
            pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        except Exception:
            traceback.print_exc()
        finally:
            shutil.rmtree(runs_dir / run_name, ignore_errors=True)


if __name__ == "__main__":
    main()
