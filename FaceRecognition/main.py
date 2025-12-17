import time
import pandas as pd
from ultralytics import YOLO
import itertools
import shutil
import os


class SimpleYOLOTrainer:

    def run(self):
        # 1. Zaktualizowany GRID (zgodny z Twoim wzorem, ale dla face_data)
        grid = {
            # Możesz dodać face_data_aug.yaml i face_data_raw.yaml jeśli istnieją
            'data': ['face_data.yaml', 'face_data_raw.yaml', 'data_aug.yaml'],
            'model': ['yolo11m.pt', 'yolo11n.pt'],
            'epochs': [100],
            'batch': [16],
            'optimizer': ['SGD', 'AdamW'],
            'lr0': [0.01, 0.001],
        }

        keys = list(grid.keys())
        values = list(grid.values())

        run_common = dict(
            save=True,  # Ważne: włączone save, żeby walidacja działała poprawnie
            plots=False,
            save_txt=False,
            save_json=False,
            save_conf=False,
            save_crop=False,
            save_period=-1,
            project='temp_runs',
            exist_ok=True
        )

        exp_id = 0
        csv_filename = 'results.csv'

        # Iteracja po gridzie
        for combo in itertools.product(*values):
            exp_id += 1
            config = dict(zip(keys, combo))

            # Unikalna nazwa folderu dla każdego runa
            run_name = f"exp_{exp_id}"

            print(f"\n[{exp_id}] Konfiguracja: {config['data']} | Model: {config['model']} | {config['optimizer']}")

            # 2. LOGIKA STEROWANIA AUGMENTACJĄ (Skopiowana z wzorca)
            aug_params = {}
            data_file_name = config['data']
            aug_info_str = "UNKNOWN"

            # Kompletna lista wyłączająca augmentację
            disable_yolo_aug = {
                'degrees': 0.0, 'translate': 0.0, 'scale': 0.0,
                'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0,
                'fliplr': 0.0, 'hsv_h': 0.0, 'hsv_s': 0.0,
                'hsv_v': 0.0, 'mosaic': 0.0, 'mixup': 0.0,
                'copy_paste': 0.0, 'erasing': 0.0, 'auto_augment': None
            }

            if 'aug' in data_file_name:
                print("   -> Wykryto 'aug'. Wyłączam augmentację YOLO.")
                aug_params = disable_yolo_aug
                aug_info_str = "Custom (Albumentations)"

            elif 'raw' in data_file_name:
                print("   -> Wykryto 'raw'. Wyłączam augmentację YOLO.")
                aug_params = disable_yolo_aug
                aug_info_str = "None (Raw Baseline)"

            else:
                print("   -> Standard. Używam augmentacji YOLO.")
                aug_params = {}
                aug_info_str = "YOLO Internal"

            model = YOLO(config['model'])
            start = time.time()

            try:
                # TRENING
                model.train(
                    data=config['data'],
                    epochs=config['epochs'],
                    imgsz=640,
                    batch=config['batch'],
                    lr0=config['lr0'],
                    optimizer=config['optimizer'],
                    patience=10,
                    verbose=False,
                    name=run_name,
                    **aug_params,
                    **run_common
                )

                # Próba odczytania epoki (best) z pliku
                best_epoch_idx = getattr(model.trainer, 'epoch', -1)

                train_time = time.time() - start

                # WALIDACJA
                metrics = model.val(
                    data=config['data'],
                    split='val',
                    save=False,
                    plots=False
                )

                # ZBIERANIE WYNIKU
                result = {
                    'exp_id': exp_id,
                    'data': config['data'],
                    'model': config['model'],
                    'best_epoch': best_epoch_idx,
                    'optimizer': config['optimizer'],
                    'lr0': config['lr0'],
                    'augmentation_type': aug_info_str,
                    'test_mAP50': float(metrics.box.map50),
                    'test_mAP50-95': float(metrics.box.map),
                    'test_precision': float(metrics.box.p.mean()),
                    'test_recall': float(metrics.box.r.mean()),
                    'fitness': float(metrics.fitness),
                    'time_min': train_time / 60
                }

                # ZAPIS DO CSV (TRYB APPEND)
                df = pd.DataFrame([result])
                write_header = not os.path.exists(csv_filename)
                df.to_csv(csv_filename, mode='a', header=write_header, index=False)

                print(f"✅ mAP: {metrics.box.map:.3f} | {train_time / 60:.1f}min")
                print("📁 Zapisano: results.csv")

                # SPRZĄTANIE (temp_runs)
                run_dir = os.path.join('temp_runs', run_name)
                if os.path.exists(run_dir):
                    shutil.rmtree(run_dir)
                print("🧹 Wyczyszczono pliki tymczasowe.")

            except Exception as e:
                print(f"❌ Błąd treningu: {e}")


if __name__ == "__main__":
    trainer = SimpleYOLOTrainer()
    trainer.run()
