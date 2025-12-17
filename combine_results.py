import pandas as pd

df_faces = pd.read_csv("results2.csv")
df_weapons = pd.read_csv("results.csv")

df_faces["dataset_name"] = "Faces"
df_weapons["dataset_name"] = "Weapons"

df_combined = pd.concat([df_faces, df_weapons], ignore_index=True)

df_combined.to_csv("results_combined.csv", index=False)
