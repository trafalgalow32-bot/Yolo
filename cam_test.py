# cam_test.py

import cv2
from ultralytics import YOLO
from datetime import datetime
import os
import json
import time
from save_json import save_frame_json
from save_json import flush_json

model = YOLO("yolov8n.pt")

streaming = True

def make_json(res):
    result = res[0]
    detection = []
    for box in result.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cid = int(box.cls[0])
        cname = result.names[cid]
        conf = float(box.conf[0])

        detection.append( {
            "class_id": cid,
            "class_name": cname,
            "confidence": round(conf, 3),
            "bbox" : {
                "x1":x1, "y1":y1, "x2":x2, "y2":y2
            }
        })

    return { "count" : len(detection), "detection": detection }

async def test_cam(request):
    global streaming
    streaming = True

    # now = datetime.now().strftime("%Y%m%d_%H%M")
    # json_path = os.path.join("records", f"cam_{now}.json")

    save_data = {}
    annotations = []
    cap = cv2.VideoCapture(0)
    save_data = {
        "info" : {
            "width": int( cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS)
        },
        "frames": annotations
    }
    frame_no = 0

    current_minute = datetime.now().strftime("%Y^%m%d_%H%M")
    json_path = os.path.join("records", f"cam_{current_minute}.jsonl")
    
    while streaming:
        if await request.is_disconnected():
            streaming=False
            break
        ret, frame = cap.read()
        if not ret: break
        res = model(frame, verbose=False)

        frame_no += 1

# 분석 결과 json 저장
        now_minute = datetime.now().strftime("%Y%m%d_%H%M")
        if now_minute != current_minute:
            flush_json(json_path)

            current_minute = now_minute
            json_path = os.path.join("records", f"cam_{now_minute}.json")
            # annotations.append( make_json( res) )          
            
            
        save_data["frames"] = make_json( res )

        save_frame_json(json_path, save_data)            

        fr = res[0].plot()

        success, buffer = cv2.imencode(
            ".jpg", fr,
            # [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        if not success: continue
        yield(
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"+
            buffer.tobytes() + b"\r\n"
        )

#     cv2.imshow("cam", fr)
#     if cv2.waitKey(1) == 27: break
# cap.release()
# cv2.destroyAllWindows()