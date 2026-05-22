# yolo_study3.py

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/cup.mp4")

sel_object = list(
    map(
        int,
        input("탐지 객체 번호 입력 ").split()
    )
)

while True:
    r, fr = cap.read()
    if not r : break
    fr = cv2.resize(fr, (480,640))

    results = model(
    fr,
    classes=sel_object
    ) # []클래스에 해당하는 객체 찾아! model.names 실행 후 터미널 창에 번호들 참고! 
    
    res = results[0].plot()
    cv2.imshow("yolo", res)
    if cv2.waitKey(30) == 27: break

cap.release()
cv2.destroyAllWindows()

# _, target_frame = cap.read()
# res = model(target_frame)
# boxs = res[0].boxes

# dec_name = []
# for box in boxs:
#     cid = int(box.cls[0])
#     dec_name.append( f"{cid}. {model.names[cid]}" )

# for name in dec_name:
#     print(name)



# names = model.names

# results = model("images/scissors1.jpg")
# results[0].show()

# print(results[0].boxes)

# find_object = []
# find = input("탐지 객체명 입력 : ")
# for k, v in names.items():
#     if v in find:
#         find_object.append(k)

# results = model.predict(
#     source="videos/cup.mp4",
#     classes=sel_object, # []클래스에 해당하는 객체 찾아! model.names 실행 후 터미널 창에 번호들 참고! 
#     imgsz = (320,640),
#     conf=0.6,
#     show=True
#     )
# # cv2.waitKey(0)
# # cv2.destoryAllWindows()

# results[0].show()

# print( model.names )