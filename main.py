import time
import pandas as pd
from ultralytics import YOLO
import itertools


class SimpleYOLOTrainer:
    def __init__(self):
        self.results = []

    def run(self):
        grid = {
            'data': ['data.yaml', 'data_aug.yaml'],
            'model': ['yolov11n.pt', 'yolov11m.pt'],
            'epochs': [100],
            'batch': [16, 32],
            'optimizer': ['SGD', 'AdamW'],
            'lr0': [0.01, 0.001],
        }

        keys = list(grid.keys())
        values = list(grid.values())

        run_common = dict(
            save=False,
            plots=False,
            save_txt=False,
            save_json=False,
            save_conf=False,
            save_crop=False,
            save_period=-1,
            project='.',
            name='',
            exist_ok=True,
            save_dir='./'
        )

        exp_id = 0
        for combo in itertools.product(*values):
            exp_id += 1
            config = dict(zip(keys, combo))

            print(f"\n[{exp_id}] Konfiguracja: {config['data']} | Model: {config['model']}")

            # 2. LOGIKA STEROWANIA AUGMENTACJĄ
            aug_params = {}

            # Jeśli w nazwie pliku jest 'aug' (czyli data_aug.yaml) -> WYŁĄCZAMY augmentację YOLO
            if 'aug' in config['data']:
                print("   -> Wykryto pre-augmentowane dane. Wyłączam augmentację YOLO.")
                aug_params = {
                    'hsv_h': 0, 'hsv_s': 0, 'hsv_v': 0,
                    'degrees': 0, 'translate': 0, 'scale': 0,
                    'shear': 0, 'perspective': 0,
                    'flipud': 0, 'fliplr': 0,
                    'mosaic': 0, 'mixup': 0,
                    'copy_paste': 0, 'erasing': 0
                }
            else:
                # Jeśli to zwykły plik (data.yaml) -> Pusty słownik = Domyślna augmentacja YOLO włączona
                print("   -> Wykryto czyste dane. Używam wewnętrznej augmentacji YOLO.")
                aug_params = {}

            model = YOLO(config['model'])
            start = time.time()

            try:
                # Przekazujemy **aug_params rozpakowane do funkcji train
                model.train(
                    data=config['data'],
                    epochs=config['epochs'],
                    imgsz=640,
                    batch=16,
                    lr0=config['lr0'],
                    optimizer=config['optimizer'],
                    patience= 10,
                    verbose=False,
                    **aug_params,
                    **run_common
                )

                train_time = time.time() - start

                metrics = model.val(data=config['data'], split='val', save=False, plots=False, save_dir='./')

                self.results.append({
                    'exp_id': exp_id,
                    'data': config['data'],
                    'model': config['model'],
                    'epochs': config['epochs'],
                    'optimizer': config['optimizer'],
                    'lr0': config['lr0'],
                    'augmentation': 'OFF (Custom Data)' if aug_params else 'ON (YOLO Internal)',
                    'test_mAP50': float(metrics.box.map50),
                    'test_mAP50-95': float(metrics.box.map),
                    'test_precision': float(metrics.box.p.mean()),
                    'test_recall': float(metrics.box.r.mean()),
                    'fitness': float(metrics.fitness),
                    'time_min': train_time / 60
                })

                df = pd.DataFrame(self.results).sort_values('test_mAP50-95', ascending=False)
                df.to_csv('results.csv', index=False)

                print(f"✅ mAP: {metrics.box.map:.3f} | {train_time / 60:.1f}min")
                print("📁 Zapisano: results.csv")

            except Exception as e:
                print(f"❌ Błąd treningu: {e}")


if __name__ == "__main__":
    trainer = SimpleYOLOTrainer()
    trainer.run()
