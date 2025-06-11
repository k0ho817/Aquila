import cv2
from ultralytics import YOLO

# 모델 로드
model = YOLO('runs/detect/aquila_v8n_epoch100/weights/best.pt')

# 입력 영상 열기
fname = "raw_video2"
input_path = f'../test_source/{fname}.mp4'
cap = cv2.VideoCapture(input_path)

# 출력 영상 설정 (640x640으로 고정)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
output_size = (640, 640)
out = cv2.VideoWriter(f'../output/output_detected_{fname}.mp4', fourcc, fps, output_size)

# 프레임 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, output_size)
    results = model(resized_frame, imgsz=640, conf=0.4)

    # 원본 프레임 복사 후 사람만 바운딩박스 시각화
    annotated = resized_frame.copy()
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)

        if cls_id == 0:  # class 0 = person
            x1, y1, x2, y2 = xyxy
            label = f'person {conf:.2f}'
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 저장
    out.write(annotated)

# 종료 처리
cap.release()
out.release()
print(f"처리 완료! 결과 영상: output_detected_{fname}.mp4")