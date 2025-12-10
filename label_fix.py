import csv
import os


def load_metadata(metadata_path):
    """
    Wczytuje mapowanie nazwa_pliku -> target_class_id z pliku CSV.
    """
    mapping = {}
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Pobieramy nazwę pliku labela i jego poprawne target id
                label_file = row['labelfile'].strip()
                target_id = row['target'].strip()
                mapping[label_file] = target_id
        print(f"Załadowano {len(mapping)} wpisów z metadata.")
        return mapping
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {metadata_path}")
        return {}


def fix_yolo_labels(directory, mapping):
    """
    Przechodzi przez pliki w folderze i poprawia klasę zgodnie z mapowaniem.
    """
    processed_count = 0
    skipped_count = 0

    # Iterujemy po wszystkich plikach w folderze
    for filename in os.listdir(directory):
        if not filename.endswith(".txt") or filename == "classes.txt":
            continue

        # Sprawdzamy czy mamy ten plik w naszych metadanych
        if filename in mapping:
            correct_class_id = mapping[filename]
            file_path = os.path.join(directory, filename)

            new_lines = []
            modified = False

            with open(file_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                # parts[0] to obecne class_id (np. 0)
                # parts[1:] to współrzędne bboxa
                current_id = parts[0]

                # Jeśli ID jest inne niż w metadanych, podmieniamy
                # (lub po prostu nadpisujemy dla pewności)
                if current_id != correct_class_id:
                    parts[0] = correct_class_id
                    new_line = " ".join(parts) + "\n"
                    new_lines.append(new_line)
                    modified = True
                else:
                    new_lines.append(line)

            # Zapisujemy plik tylko jeśli wprowadzono zmiany
            if modified:
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)
                processed_count += 1
            else:
                # Plik był już poprawny
                pass

        else:
            # Plik txt istnieje w folderze, ale nie ma go w metadata.csv
            # print(f"Pominięto (brak w metadata): {filename}")
            skipped_count += 1

    print(f"\nZakończono.")
    print(f"Poprawiono plików: {processed_count}")
    print(f"Pominięto plików (brak w metadata lub brak zmian): {skipped_count}")


if __name__ == "__main__":
    # Ustawienia
    METADATA_FILE = 'metadata.csv'  # Ścieżka do Twojego pliku metadata
    LABELS_DIR = os.path.join('weapon_detection', 'train', 'labels')

    # Uruchomienie
    class_mapping = load_metadata(METADATA_FILE)
    if class_mapping:
        fix_yolo_labels(LABELS_DIR, class_mapping)
