import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (12, 8),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12
})
sns.set_theme(style="whitegrid", context="notebook", palette="muted")

CSV_PATH = "results_combined.csv"
OUT_DIR = Path("analysis_out")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def load_and_prep_data(path=CSV_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    num_cols = ["test_mAP50", "test_mAP50-95", "test_precision", "test_recall",
                "time_min", "best_epoch", "fitness"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "augmentation_type" in df.columns:
        df["Augmentation"] = df["augmentation_type"]
    elif "augmentation" in df.columns:
        df["Augmentation"] = df["augmentation"]
    elif "aug_mode" in df.columns:
        df["Augmentation"] = df["aug_mode"]
    else:
        df["Augmentation"] = "Unknown"

    aug_map = {
        'Custom (Albumentations)': 'Custom',
        'None (Raw Baseline)': 'None',
        'YOLO Internal': 'YOLO'
    }
    df["Aug_Short"] = df["Augmentation"].replace(aug_map)
    df.rename(columns={"best_epoch": "stopped_epoch"}, inplace=True)
    return df

def save_plot(name):
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", bbox_inches='tight')
    plt.close()
    print(f" -> Zapisano: {name}.png")

def main():
    print("=== START ANALYSIS ===")
    df = load_and_prep_data()
    print(f"Dane: {len(df)} wierszy.")

    baseline_means = df[df["Aug_Short"] == "None"].groupby("dataset_name")["test_mAP50-95"].mean()

    def calculate_gain(row):
        base = baseline_means.get(row["dataset_name"], 0)
        return row["test_mAP50-95"] - base

    df["Aug_Gain"] = df.apply(calculate_gain, axis=1)
    plot_data = df[df["Aug_Short"] != "None"].copy()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=plot_data, x="dataset_name", y="Aug_Gain", hue="Aug_Short",
        estimator=np.mean, errorbar=None, palette=["#e74c3c", "#3498db"]
    )
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
    plt.title("Zysk mAP z Augmentacji (względem 'None' = 0)")
    plt.ylabel("Zmiana mAP (Delta)")
    for container in ax.containers:
        ax.bar_label(container, fmt=' {:+.3f}', padding=3, fontsize=10)
    save_plot("1_augmentation_gain_delta")

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="dataset_name", y="test_mAP50-95", hue="Aug_Short", palette="Set2")
    plt.title("Rozkład wyników mAP50-95 (Faces vs Weapons)")
    save_plot("2_dataset_distribution")

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df, x="model", y="fitness", hue="Aug_Short",
        estimator=np.mean, errorbar="sd", palette="magma"
    )
    plt.title("Ranking Fitness Score (Ogólna jakość modelu)")
    save_plot("3_fitness_score_ranking")

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df, x="optimizer", y="test_mAP50-95", hue="lr0", palette="Set2"
    )
    plt.title("Stabilność Optimizerów: Rozrzut wyników mAP")
    save_plot("4_optimizer_stability")

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df, x="time_min", y="test_mAP50-95",
        hue="Aug_Short", style="model", size="stopped_epoch",
        sizes=(50, 200), alpha=0.8, palette="deep"
    )
    plt.title("Efektywność: Czas Treningu vs Jakość (mAP)")
    plt.xlabel("Czas treningu (min)")
    save_plot("5_efficiency_time_vs_map")

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df, x="dataset_name", y="time_min", hue="Aug_Short",
        estimator=np.mean, errorbar="sd", palette="Reds"
    )
    plt.title("Średni Czas Treningu wg Metody Augmentacji")
    plt.ylabel("Czas (minuty)")
    save_plot("6_training_time_comparison")

    print(f"\nAnaliza zakończona. Sprawdź folder: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
