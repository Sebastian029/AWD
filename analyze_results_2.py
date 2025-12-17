import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ustawienia estetyczne
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
                "time_min", "best_epoch"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ujednolicenie augmentacji
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

    # Zmiana nazwy kolumny dla jasności (to epoka zatrzymania)
    df.rename(columns={"best_epoch": "stopped_epoch"}, inplace=True)

    return df


def save_plot(name):
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}.png", bbox_inches='tight')
    plt.close()
    print(f" -> Zapisano: {name}.png")


def main():
    print("=== START OPTIMIZED ANALYSIS ===")
    df = load_and_prep_data()
    print(f"Dane wczytane: {len(df)} eksperymentów.")

    # --- 1. INTERACTION PLOT: CZYSTY ZYSK Z AUGMENTACJI ---
    print("Generowanie: Zysk z Augmentacji (Interaction Plot)...")

    # Obliczamy średnią dla 'None' w każdym zbiorze jako punkt odniesienia
    baseline_means = df[df["Aug_Short"] == "None"].groupby("dataset_name")["test_mAP50-95"].mean()

    # Funkcja obliczająca zysk (delta)
    def calculate_gain(row):
        base = baseline_means.get(row["dataset_name"], 0)
        return row["test_mAP50-95"] - base

    df["Aug_Gain"] = df.apply(calculate_gain, axis=1)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df, x="dataset_name", y="Aug_Gain", hue="Aug_Short",
        estimator=np.mean, errorbar=None, palette="coolwarm_r"
    )
    plt.axhline(0, color='black', linewidth=1)
    plt.title("Zysk (lub strata) mAP z Augmentacji względem braku augmentacji (Baseline)")
    plt.ylabel("Zmiana mAP (Delta)")
    save_plot("1_augmentation_gain_interaction")

    # --- 2. STABILNOŚĆ OPTIMIZERÓW (BOXPLOT) ---
    print("Generowanie: Stabilność Optimizerów...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df, x="optimizer", y="test_mAP50-95", hue="lr0",
        palette="Set2", showfliers=True
    )
    plt.title("Stabilność Optimizerów: Rozrzut wyników mAP")
    plt.ylabel("mAP 50-95")
    save_plot("2_optimizer_stability_boxplot")

    # --- 3. KRZYWA UCZENIA (STOPPED EPOCH vs LR) ---
    print("Generowanie: Analiza czasu zbieżności...")
    plt.figure(figsize=(10, 6))
    # Boxplot pokazuje rozkład, Stripplot pokazuje konkretne punkty
    sns.boxplot(
        data=df, x="optimizer", y="stopped_epoch", hue="lr0",
        dodge=True, boxprops={'alpha': 0.4}
    )
    sns.stripplot(
        data=df, x="optimizer", y="stopped_epoch", hue="lr0",
        dodge=True, marker="D", size=7, alpha=0.8
    )
    plt.title("Wpływ LR na szybkość zbieżności (Early Stopping)")
    plt.ylabel("Epoka zatrzymania treningu")
    save_plot("3_convergence_stopped_epoch")

    # --- 4. EFEKTYWNOŚĆ (SCATTER) - ZMODYFIKOWANY ---
    print("Generowanie: Czas vs Wynik...")
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df, x="time_min", y="test_mAP50-95",
        hue="Aug_Short", style="model", size="stopped_epoch",
        sizes=(50, 200), alpha=0.8, palette="deep"
    )
    plt.title("Koszt vs Zysk: Czy dłuższy trening (czas/epoki) daje lepsze mAP?")
    plt.xlabel("Czas treningu (minuty)")
    save_plot("4_efficiency_time_vs_map")

    # --- 5. PRECYZJA VS RECALL (DLA BEZPIECZEŃSTWA) ---
    print("Generowanie: Precision vs Recall...")
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df, x="test_recall", y="test_precision",
        hue="dataset_name", style="model", size="test_mAP50-95",
        sizes=(50, 250), alpha=0.8
    )
    plt.title("Charakterystyka błędów: Co poświęcamy? (Precision vs Recall)")
    plt.axvline(x=0.5, ls='--', c='grey', alpha=0.3)
    plt.axhline(y=0.5, ls='--', c='grey', alpha=0.3)
    save_plot("5_precision_vs_recall")

    # --- 6. TOP CONFIGS TABLE ---
    print("\nGenerowanie tabeli TOP 10...")
    cols = ["dataset_name", "model", "Aug_Short", "optimizer", "lr0", "test_mAP50-95", "stopped_epoch", "time_min"]
    top10 = df.sort_values("test_mAP50-95", ascending=False).head(10)[cols]
    top10.to_csv(OUT_DIR / "top_10_optimized.csv", index=False)
    print(top10.to_string(index=False))

    print(f"\nAnaliza zakończona. Wyniki: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
