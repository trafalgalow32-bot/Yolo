# yolo_track_red.py

import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def tracking_frame():
    cap = cv2.VideoCapture("videos/track_car.mp4")

    box_color = [
        (22, 219, 29),
        (229, 209, 92),
        (95, 95, 241), 
        (220, 97, 243),
        (0, 228, 255)
    ]

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (640, 480) )

    # persist : 이전 프레임의 추적 정보를 계속 사용
    # yolo모델로 분석한 결과 results에 저장함
    # results는 리스트 구조이다. 각 인덱스에 분석한 이미지 한장 씩 들어 있다. 
    # results[0] 내부 값
    # boxes : 탐지된 객체 정보, masks : 객체 마스크
    # names : 객체 이름 (딕셔너리), orig_img: 원본이미지
    # path : 이미지 경로, speed : 처리 시간
    # probs : 분류 확률 정보

        results = model.track(
            frame, conf = 0.4, persist = True
            )

        boxs = results[0].boxes
        u_class_ids = sorted(
            set( int(box.cls[0]) for box in boxs)
        )
        idex_map = {
            class_id : index
            for index, class_id in enumerate(u_class_ids)
        }

        for box in boxs:
    # xywh는 첫번째 좌표와 바운딩박스의 너비, 높이
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cid = int(box.cls[0])
            cname = results[0].names[cid]
            track_id = int(box.id[0])
            cv2.rectangle(
                frame, (x1,y1), (x2,y2),
                box_color[idex_map[cid]], 2
            )

        cv2.putText(
            frame, f"{cname} ({track_id})",
            (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, box_color[idex_map[cid]], 2
        )

        # 브라우저로 스트리밍 하기 위한 코드
        success, buffer = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        if not success: continue
        yield(
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"+
            buffer.tobytes() + b"\r\n"
        )
        
    #     cv2.imshow("car", frame)
    #     if cv2.waitKey(30) == 27: break

    # cap.release()
    # cv2.destroyAllWindows()