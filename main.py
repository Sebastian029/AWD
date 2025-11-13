import time
import pandas as pd
from ultralytics import YOLO
import itertools

class SimpleYOLOTrainer:
    def __init__(self):
        self.results = []

    def run(self):
        grid = {
            'data': ['data.yaml'],
            'model': ['yolov8n.pt'],
            'epochs': [2,5],
            'optimizer': ['SGD', 'Adam'],
            'lr0': [0.01],
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

            print(f"\n[{exp_id}] {config}")

            aug_params = {}
            if 'data_aug' in config['data']:
                aug_params = {'hsv_h': 0, 'hsv_s': 0, 'hsv_v': 0, 'degrees': 0,
                              'translate': 0, 'scale': 0, 'mosaic': 0, 'mixup': 0}

            model = YOLO(config['model'])
            start = time.time()

            try:
                model.train(
                    data=config['data'],
                    epochs=config['epochs'],
                    imgsz=640,
                    batch=16,
                    lr0=config['lr0'],
                    optimizer=config['optimizer'],
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
                    'augmentation': 'OFF' if aug_params else 'ON',
                    'test_mAP50': float(metrics.box.map50),
                    'test_mAP50-95': float(metrics.box.map),
                    'test_precision': float(metrics.box.p.mean()),
                    'test_recall': float(metrics.box.r.mean()),
                    'time_min': train_time / 60
                })

                df = pd.DataFrame(self.results).sort_values('test_mAP50-95', ascending=False)
                df.to_csv('results.csv', index=False)

                print(f"✅ mAP: {metrics.box.map:.3f} | {train_time / 60:.1f}min")
                print("📁 Zapisano: results.csv")

            except Exception as e:
                print(f"❌ {e}")

if __name__ == "__main__":
    trainer = SimpleYOLOTrainer()
    trainer.run()
