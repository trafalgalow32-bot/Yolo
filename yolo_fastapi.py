# yolo_fastapi.py

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request

from yolo_track_red import tracking_frame
from cam_test import test_cam

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/webcam")
async def get_cam(request: Request):
    return StreamingResponse(
        test_cam(request) ,
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
@app.get("/")
def home():
    return {"message" : "ok"}

@app.get("/video")
def video():
    return StreamingResponse(
        tracking_frame(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )