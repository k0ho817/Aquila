from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='/home/mkh99/Aquila/data/Aquila-1/data.yaml',
    epochs=100,
    imgsz=960,
    batch=128,
    name='aquila_v8n_epoch100_2gpu_3090',
    patience=10,
    device='0,1',
    workers=8,
    amp=True
)