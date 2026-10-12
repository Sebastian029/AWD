"""Plots comparing grid-search results (results/*_grid.csv -> results/plots/).

NOTE: weapons grid results were measured on a val split where 77/143 images are copies of train images,
so absolute weapons numbers are inflated; see src/check_dataset.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from combine_results import combine

CSV_PATH = Path("results/combined.csv")
OUT_DIR = Path("results/plots")

plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12})
sns.set_theme(style="whitegrid", context="notebook", palette="muted")

AUG_SHORT = {"Custom (Albumentations)": "Custom", "None (Raw Baseline)": "None", "YOLO Internal": "YOLO"}


def load_data(path=CSV_PATH):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Aug_Short"] = df["augmentation_type"].replace(AUG_SHORT)
    return df


def save_plot(name):
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    plt.close()
    print(f" -> {name}.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combine()  # always rebuild combined.csv from the per-task CSVs
    df = load_data()
    print(f"{len(df)} runs loaded")

    baseline = df[df["Aug_Short"] == "None"].groupby("dataset_name")["val_mAP50-95"].mean()
    df["Aug_Gain"] = df["val_mAP50-95"] - df["dataset_name"].map(baseline).fillna(0)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df[df["Aug_Short"] != "None"], x="dataset_name", y="Aug_Gain", hue="Aug_Short",
                     estimator=np.mean, errorbar=None, palette=["#e74c3c", "#3498db"])
    plt.axhline(0, color="black", linewidth=1.5, linestyle="--")
    plt.title("mAP gain from augmentation (relative to no augmentation)")
    plt.ylabel("mAP50-95 delta")
    for container in ax.containers:
        ax.bar_label(container, fmt=" {:+.3f}", padding=3, fontsize=10)
    save_plot("1_augmentation_gain_delta")

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="dataset_name", y="val_mAP50-95", hue="Aug_Short", palette="Set2")
    plt.title("mAP50-95 distribution per dataset")
    save_plot("2_dataset_distribution")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="model", y="fitness", hue="Aug_Short", estimator=np.mean, errorbar="sd",
                palette="magma")
    plt.title("Fitness score by model and augmentation")
    save_plot("3_fitness_score_ranking")

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
