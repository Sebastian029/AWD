"""Merge the per-task grid-search CSVs into results/combined.csv."""
import pandas as pd

from common import TASKS


def combine(out="results/combined.csv"):
    frames = []
    for name, t in TASKS.items():
        df = pd.read_csv(f"results/{name}_grid.csv")
        df["dataset_name"] = t["label"]
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)


if __name__ == "__main__":
    combine()
