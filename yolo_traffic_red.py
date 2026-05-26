# yolo_traffic_red.py

import cv2
import numpy as np

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# 차량 주행 중 빨간 신호등 발견되면 멈추라고 경고 하기
# 영상에서 특정 색상을 구별 하기

cap = cv2.VideoCapture("videos/traffic_red.mp4")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret : break
    
    frame_count += 1

    frame = cv2.resize(frame, (480, 640) )

    results = model(frame, classes=[9], conf = 0.2)

    boxs = results[0].boxes
    red_detect = False

    for box in boxs:
        x1,y1,x2,y2 = map(int, box.xyxy[0])

        roi = frame[y1:y2, x1:x2] # 프레임에서 신호등 영역만 가져오기

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0,120,70])
        up_red1 = np.array([10,255,255])

        lower_red2 = np.array([170,120,70])
        up_red2 = np.array([180,255,255])

        mask1 = cv2.inRange(hsv, lower_red1, up_red1)
        mask2 = cv2.inRange(hsv, lower_red2, up_red2)
        red_mask = mask1 + mask2
        red_pixel = cv2.countNonZero(red_mask)
        if red_pixel > 50:
            red_detect = True
            cv2.rectangle(frame, (x1,y1), (x2,y2), 
                          (0,0,255), 3)
    
    if red_detect:
        blink_on = (frame_count // 10) % 2 == 0

        if blink_on:
            red_overay = np.zeros_like(frame)
            red_overay[:] = (0,0,255)

            frame = cv2.addWeighted(
                frame, 0.65, red_overay, 0.35, 0
            )
            cv2.putText(frame, "STOP!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 8)
    cv2.imshow("red", frame)
    if cv2.waitKey(30) == 27: break
cap.release()
cv2.destoryAllWindows()
