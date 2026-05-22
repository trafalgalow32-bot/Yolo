# yolo_study1.py

from ultralytics import YOLO
import cv2

# ultralytics yolo 모델 크기
# n : 가장 가벼움, s : n보다 조금 더 
# m : 중간크기 pc성능좋아야함, 
# l : 많은 학습이 되어 있음 - m모델의 5배?
# x : 최상위 모델, AI전용 서버에서 운영 가능

model = YOLO("yolov8n.pt") # 8버젼 n(나노)
# model = YOLO("yolov8s.pt") # 8버젼 s

# results = model( "videos/bird.mp4")
# results = model( "images/bird.png") # picture
# results[0].show()

# print(model.names)

cap = cv2.VideoCapture("videos/bird.mp4")

while True:
    ret, frame = cap.read()
    if not ret :break
    frame = cv2.resize(frame, (480, 640))

    result = model(frame)

    anno = result[0].plot()

    cv2.imshow("bird", anno)
    
    key = cv2.waitKey(30)
    if key == 27: break
cap.release()
cv2.destroyAllWindows()