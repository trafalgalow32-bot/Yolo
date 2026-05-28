# webcam_service.py

import cv2
import os
from datetime import datetime
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

streaming = True

def stop_record():
    global streaming
    streaming = False

def record_frames():
    global streaming
    streaming = True

    cap = cv2.VideoCapture(0)

# 영상 저장은 cv2.VideoWriter 함수 사용
# 영상 저장하려면 저장파일 이름, 확장자, fps, 영상크기
    width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join("videos", f"cam_{now}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        video_path, fourcc, 10, (width, height) # fps 를 10으로 수정
    )

    while streaming:
        ret, frame = cap.read()
        if not ret:break

        results = model(frame, verbose=False)
        res = results[0].plot()

        writer.write(res) # 영상저장 - 프레임(이미지) 하나씩 저장

        success, buffer = cv2.imencode(
            ".jpg", res,
            # [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        if not success: continue
        yield(
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"+
            buffer.tobytes() + b"\r\n"
        )

    writer.release()
    cap.release()
    print("종료")