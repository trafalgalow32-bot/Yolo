# yolo_study5.py

import cv2
import time

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
print(model.names)
cap = cv2.VideoCapture("videos/pedestrian1.mp4")

person_hg = 1.72 # 사람의 키
focal = 400 # 카메라 초점거리

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.resize(frame, (640, 480))

    results = model(frame, classes=[0,2])

    boxs = results[0].boxes

    for box in boxs:
        x1,y1,x2,y2 = map(int, box.xyxy[0])

        dist = ( person_hg * focal ) / (y2-y1)
    
        cv2.rectangle(
            frame, (x1, y1), (x2, y2), (0,0,255), 2
        )
        cv2.putText(
            frame, f"{dist : .1f}m", (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2
        )
        if dist >= 10 :
            blink = int(time.time() * 4) % 2
            if blink == 0:
                red_overlay = frame.copy()
                red_overlay[:,:] = (0,0,255)

                frame = cv2.addWeighted(
                    frame, 0.65, red_overlay, 0.35, 0
                )

    cv2.imshow("frame", frame)
    if cv2.waitKey(30) == 27: break

cap.release()
cv2.destroyAllWindows()
        