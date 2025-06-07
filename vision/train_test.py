# train_yolo.py
from ultralytics import YOLO

model = YOLO('yolov8x.pt')

model.train(
    data='human-uav-2--3/data.yaml',
    epochs=100,
    imgsz=1280,
    batch=32,
    name='uav_person_detector_manualddp',
    patience=10,
    device=0  # 요건 무시됨, torchrun이 자동 분산
)