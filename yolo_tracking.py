# yolo_tracking.py

import cv2

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/phone.mp4")

while True:
    ret, frame = cap.read()
    if not ret : break

    frame = cv2.resize(frame, (480, 640))

    results = model.track( frame, persist = True, classes=[0], 
                          tracker="botsort.yaml")
    # 사람 추적시 tracker에 bytetrack.yaml, botsort.yaml

    res = results[0].plot()
    cv2.imshow("tracking", res)
    if cv2.waitKey(30) == 27 : break

cap.release()
cv2.destroyAllWindows()