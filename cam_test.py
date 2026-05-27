# cam_test.py

import cv2
from ultralytics import YOLO
from datetime import datetime
import os
import json

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

    now = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = os.path.join("records", f"cam_{now}.json")

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
    while streaming:
        if await request.is_disconnected():
            streaming=False
            break
        ret, frame = cap.read()
        if not ret: break
        res = model(frame, verbose=False)

# 분석 결과 json 저장
        # annotations.append( make_json( res) )
        save_data["frames"] = make_json( res )

        # now = datetime.now().strftime("%Y%m%d_%H%M")
        # json_path = os.path.join("records", f"cam_{now}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                save_data, f, ensure_ascii=False, indent=4
            )   

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