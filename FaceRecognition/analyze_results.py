# analyze_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams.update({"figure.dpi": 140})
sns.set(style="whitegrid")

CSV_PATH = "results.csv"
OUT_DIR = Path("analysis_out")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def load_data(path=CSV_PATH):
    df = pd.read_csv(path)
    # Uporządkuj nazwy kolumn (czasem różne separatory)
    df.columns = [c.strip() for c in df.columns]
    # Typy liczbowe
    num_cols = ["epochs", "lr0", "test_mAP50", "test_mAP50-95", "test_precision", "test_recall", "time_min"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Porządkuj kategorie
    for c in ["model", "optimizer", "data", "augmentation"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

def save_table(df, name):
    df.round(4).to_csv(OUT_DIR / f"{name}.csv", index=False)

def summarize_overall(df):
    agg = df.agg({
        "test_mAP50-95": ["mean", "std", "max"],
        "test_mAP50": ["mean", "std", "max"],
        "test_precision": ["mean", "std"],
        "test_recall": ["mean", "std"],
        "time_min": ["mean", "sum"]
    })
    agg.to_csv(OUT_DIR / "overall_summary.csv")

def group_stats(df, by, metrics=("test_mAP50-95","test_mAP50","test_precision","test_recall","time_min")):
    g = df.groupby(by, dropna=False)[list(metrics)].agg(["mean","std","count"]).reset_index()
    # Spłaszcz kolumny MultiIndex
    g.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in g.columns]
    save_table(g, f"group_{'_'.join(by)}")
    return g

def top_k(df, k=10, sort_by="test_mAP50-95"):
    t = df.sort_values(sort_by, ascending=False).head(k)
    save_table(t, f"top_{k}_{sort_by}")
    return t

def plot_bar(df, x, y, hue=None, title="", fname="plot.png", rotate=False):
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=df, x=x, y=y, hue=hue, estimator=np.mean, errorbar="sd")
    ax.set_title(title)
    if rotate:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname)
    plt.close()

def plot_rel(df, x, y, hue=None, style=None, title="", fname="rel.png"):
    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(data=df, x=x, y=y, hue=hue, style=style, marker="o")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname)
    plt.close()

def plot_scatter(df, x, y, hue=None, size=None, title="", fname="scatter.png"):
    plt.figure(figsize=(6, 4))
    ax = sns.scatterplot(data=df, x=x, y=y, hue=hue, size=size, sizes=(40, 200), alpha=0.8)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname)
    plt.close()

def main():
    df = load_data()

    # Filtry jakości (opcjonalnie)
    df = df.dropna(subset=["test_mAP50-95", "test_mAP50"])
    save_table(df, "clean_results")

    # Podstawowe podsumowania
    summarize_overall(df)
    group_stats(df, ["model"])
    group_stats(df, ["optimizer"])
    group_stats(df, ["epochs"])
    group_stats(df, ["lr0"])
    group_stats(df, ["model", "optimizer"])
    top_k(df, k=min(10, len(df)), sort_by="test_mAP50-95")

    # Wykresy porównawcze
    # 1) Model vs mAP50-95 (średnia + SD), z podziałem na optimizer
    plot_bar(
        df, x="model", y="test_mAP50-95", hue="optimizer",
        title="Model vs mAP50-95 by Optimizer",
        fname="model_map95_by_optimizer.png", rotate=True
    )

    # 2) Optimizer vs mAP50-95
    plot_bar(
        df, x="optimizer", y="test_mAP50-95", hue=None,
        title="Optimizer vs mAP50-95",
        fname="optimizer_map95.png", rotate=False
    )

    # 3) Epoki a mAP50-95 (osobno dla optimizer)
    plot_rel(
        df, x="epochs", y="test_mAP50-95", hue="optimizer", style="model",
        title="Epochs vs mAP50-95",
        fname="epochs_vs_map95.png"
    )

    # 4) LR a mAP50-95 (kolor: optimizer)
    plot_scatter(
        df, x="lr0", y="test_mAP50-95", hue="optimizer", size="epochs",
        title="LR vs mAP50-95",
        fname="lr_vs_map95.png"
    )

    # 5) Precision vs Recall (kolor: konfiguracja)
    cfg_col = "config"
    df[cfg_col] = (
        df["model"].astype(str) + " | " +
        df["optimizer"].astype(str) + " | ep=" +
        df["epochs"].astype(str) + " | lr=" +
        df["lr0"].astype(str)
    )
    plot_scatter(
        df, x="test_recall", y="test_precision", hue="optimizer", size="epochs",
        title="Precision vs Recall",
        fname="precision_vs_recall.png"
    )

    # 6) Czas vs mAP50-95 (efektywność)
    plot_scatter(
        df, x="time_min", y="test_mAP50-95", hue="optimizer", size="epochs",
        title="Time (min) vs mAP50-95",
        fname="time_vs_map95.png"
    )

    print(f"Gotowe. Wyniki i wykresy w: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
