# yolo_study4.py

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/walk.mp4")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (640, 480))

    results = model(frame, classes=[0], conf = 0.6)

    boxs = results[0].boxes

    danger_rect = (0,240, 100,400)
    cv2.rectangle(
        frame, (danger_rect[0], danger_rect[1]) ,
        (danger_rect[2], danger_rect[3]), (0,228,255), 2
    )

    person_count = 0

    for box in boxs:
        person_count+=1
        x1,y1, x2,y2 = map(int, box.xyxy[0])

        cx = (x1+x2) // 2
        cy = (y1+y2) // 2
        if cx > danger_rect[0] and cx <= danger_rect[2]:
            if cy > danger_rect[1] and cy <= danger_rect[3]:
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0,0,255), 2
                )
                cv2.putText( # 각 사람 위에 번호넣기
                    frame, f"pc : {person_count}",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,255), 2
                )

    cv2.putText( # 화면에 총 몇명 찾았는지 표시
        frame, f"Total_Person : {person_count}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0,0,255), 2    
    )
    cv2.imshow("frame", frame)
    if cv2.waitKey(30) == 27: break

cap.release()
cv2.destroyAllWindows()
        