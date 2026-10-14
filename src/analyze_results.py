"""Plots comparing grid-search results.

    python src/analyze_results.py                         # weapons only -> results/plots/weapons/
    python src/analyze_results.py --tasks weapons masks   # joint comparison -> results/plots/joint/

Caveats: every grid cell is a single run (no repeated seeds); YOLO-internal runs often reach the 100-epoch cap
while the others early-stop at ~30-60 epochs, so augmentation and training length are confounded.
Weapons runs are validated on val_clean (the Kaggle val images without a copy in train); see src/make_clean_val.py.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import TASKS

OUT_DIR = Path("results/plots")

plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12})
sns.set_theme(style="whitegrid", context="notebook", palette="muted")

AUG_SHORT = {"Custom (Albumentations)": "Custom", "None (Raw Baseline)": "None", "YOLO Internal": "YOLO"}


def load_data(tasks):
    frames = []
    for name in tasks:
        df = pd.read_csv(f"results/{name}_grid.csv")
        df["dataset_name"] = TASKS[name]["label"]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["Aug_Short"] = df["augmentation_type"].replace(AUG_SHORT)
    return df


def save_plot(name):
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    plt.close()
    print(f" -> {name}.png")


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", choices=list(TASKS), default=["weapons"])
    args = ap.parse_args()
    OUT_DIR = Path("results/plots") / ("joint" if len(args.tasks) > 1 else args.tasks[0])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(args.tasks)
    print(f"{len(df)} runs loaded")

    # Best run of the grid per augmentation: means are dominated by diverged runs (e.g. AdamW, lr0=0.01).
    best = df.groupby(["dataset_name", "Aug_Short"])["val_mAP50-95"].max().reset_index()
    base = best[best["Aug_Short"] == "None"].set_index("dataset_name")["val_mAP50-95"]
    best["Gain"] = best["val_mAP50-95"] - best["dataset_name"].map(base)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=best[best["Aug_Short"] != "None"], x="dataset_name", y="Gain", hue="Aug_Short",
                     palette=["#e74c3c", "#3498db"])
    plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
    plt.title("Best-run mAP50-95 gain from augmentation (relative to no augmentation)")
    plt.ylabel("mAP50-95 delta (best run per group)")
    for container in ax.containers:
        ax.bar_label(container, fmt=" {:+.3f}", padding=3, fontsize=10)
    save_plot("1_augmentation_gain_delta")

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="dataset_name", y="val_mAP50-95", hue="Aug_Short", palette="Set2")
    plt.title("mAP50-95 distribution per dataset")
    save_plot("2_dataset_distribution")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="model", y="val_mAP50-95", hue="Aug_Short", estimator=np.mean, errorbar="sd",
                palette="magma")
    plt.title("Mean mAP50-95 by model and augmentation (error bars: SD over the grid)")
    save_plot("3_map_by_model_and_augmentation")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="optimizer", y="val_mAP50-95", hue="lr0", palette="Set2")
    plt.title("Optimizer stability: spread of mAP50-95")
    save_plot("4_optimizer_stability")

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x="time_min", y="val_mAP50-95", hue="Aug_Short", style="model",
                    size="stopped_epoch", sizes=(50, 200), alpha=0.8, palette="deep")
    plt.title("Efficiency: training time vs mAP50-95")
    plt.xlabel("Training time (min)")
    save_plot("5_efficiency_time_vs_map")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="dataset_name", y="time_min", hue="Aug_Short", estimator=np.mean, errorbar="sd",
                palette="Reds")
    plt.title("Mean training time by augmentation method")
    plt.ylabel("Time (min)")
    save_plot("6_training_time_comparison")


if __name__ == "__main__":
    main()
