# yolo_study6.py

import cv2

from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture("videos/pedstiran2.mp4")

obj_color = { "person" : (0,228,255),
             "car" : (0,0,255) }


results = model("images/dog.png")

while True:
    ret, frame = cap.read()
    if not ret : break
    frame = cv2.resize( frame, (640, 480))

    results = model(frame, classes=[0,2])

    result = results[0]

    overlay = frame.copy()
    if result.masks is not None:
        masks = result.masks.data
        boxs = result.boxes

        for mask, box in zip(masks, boxs):
            cid = int(box.cls[0])
            cname = model.names[cid]
            color = obj_color[cname]

            mask = mask.cpu().numpy()

            mask = cv2.resize( mask, (frame.shape[1], frame.shape[0]))
            mask = mask > 0.5

            overlay[mask] = color
    
    frame = cv2.addWeighted( frame )

    cv2.imshow("seg", frame)
    if cv2.waitKey(2) == 27: break

cap.release()
cv2.destroyAllWindows()
