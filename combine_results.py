import pandas as pd

# 1. Wczytaj swoje pliki (podaj poprawne ścieżki)
df_faces = pd.read_csv("results2.csv")   # Tu wpisz nazwę 1. pliku
df_weapons = pd.read_csv("results.csv") # Tu wpisz nazwę 2. pliku

# 2. Dodaj kolumnę identyfikującą zbiór
df_faces["dataset_name"] = "Faces"
df_weapons["dataset_name"] = "Weapons"

# 3. Połącz oba zbiory (jeden pod drugim)
df_combined = pd.concat([df_faces, df_weapons], ignore_index=True)

# 4. Zapisz wynik do nowego pliku
df_combined.to_csv("results_combined.csv", index=False)

print(f"Gotowe! Połączono {len(df_faces)} wierszy (Faces) i {len(df_weapons)} wierszy (Weapons).")
print(f"Razem: {len(df_combined)} wierszy. Zapisano w 'results_combined.csv'.")
