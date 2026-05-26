# yolo_study7.py

import cv2
import numpy as np
import mss

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

sct =  mss.mss()

monitor = {"top" : 100, "left": 100 , "width" : 400, "height": 300}

while True:
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    results = model(frame)
    res = results[0].plot()
    cv2.imshow("f", res)
    if cv2.waitKey(30) == 27: break

cv2.destroyAllWindows()