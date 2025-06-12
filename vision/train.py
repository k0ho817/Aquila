from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(
    data='../data/Aquila-1/data.yaml',
    epochs=100,
    imgsz=640,
    batch=96,
    name='aquila_v8s_epoch100_pt10',
    patience=10,
    device='0,1'
)