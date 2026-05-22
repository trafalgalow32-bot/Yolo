# yolo_study2.py

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/bird.mp4")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (480, 640))

    results = model(frame, conf=0.5)

    boxs = results[0].boxes

    for box in boxs:
        class_id = int(box.cls[0]) # 탐지객체의 번호
        class_name = model.names[class_id] # 탐지객체의 이름
        confidence = float( box.conf[0]) # 탐지 신뢰도

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(
            frame, (x1,y1), (x2,y2), (0,0,255), 2
        )
        cv2.putText(
            frame, f"{class_name} {confidence:.2f}",
            (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255,0,0), 2
        )
    
    cv2.imshow("result", frame)
    if cv2.waitKey(30) == 27: break
cap.release()
cv2.destroyAllWindows()