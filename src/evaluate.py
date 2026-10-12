"""Evaluate trained weights on a chosen split and print per-class AP.

    python src/evaluate.py --task masks   --split test
    python src/evaluate.py --task weapons --split val --weights path/to/best.pt
"""
import argparse

from ultralytics import YOLO

from common import config_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["weapons", "masks"], default="weapons")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--weights", help="default: runs/<task>/final_yolo11m/weights/best.pt")
    ap.add_argument("--data", help="default: configs/<task>/data.yaml")
    args = ap.parse_args()

    weights = args.weights or f"runs/{args.task}/final_yolo11m/weights/best.pt"
    data = args.data or str(config_path(args.task, "data"))
    m = YOLO(weights).val(data=data, split=args.split, plots=False, save=False)

    print(f"\n== {args.task} | {data} | split={args.split} | {weights} ==")
    print(f"mAP@50: {m.box.map50:.4f}   mAP@50-95: {m.box.map:.4f}   P: {m.box.mp:.4f}   R: {m.box.mr:.4f}")
    for i, name in m.names.items():
        if i in m.box.ap_class_index:
            print(f"  {name:18s} AP50-95 = {m.box.maps[i]:.3f}")


if __name__ == "__main__":
    main()
