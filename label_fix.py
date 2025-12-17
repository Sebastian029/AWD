import csv
import os


def load_metadata(metadata_path):

    mapping = {}
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_file = row['labelfile'].strip()
                target_id = row['target'].strip()
                mapping[label_file] = target_id
        return mapping
    except FileNotFoundError:
        return {}


def fix_yolo_labels(directory, mapping):
    processed_count = 0
    skipped_count = 0

    for filename in os.listdir(directory):
        if not filename.endswith(".txt") or filename == "classes.txt":
            continue

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

                current_id = parts[0]

                if current_id != correct_class_id:
                    parts[0] = correct_class_id
                    new_line = " ".join(parts) + "\n"
                    new_lines.append(new_line)
                    modified = True
                else:
                    new_lines.append(line)

            if modified:
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)
                processed_count += 1
            else:
                pass

        else:
            skipped_count += 1



if __name__ == "__main__":
    METADATA_FILE = 'metadata.csv'
    LABELS_DIR = os.path.join('weapon_detection', 'val', 'labels')

    class_mapping = load_metadata(METADATA_FILE)
    if class_mapping:
        fix_yolo_labels(LABELS_DIR, class_mapping)
