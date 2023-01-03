import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import cv2
import mediapipe as mp
import time


class faceDetector():
    def __init__(self, mode = False, maxHands=2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpFaces = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.Faces = self.mpFaces.FaceDetection()
        

    def findFaces(self, img, w, h):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.Faces.process(imgRGB)
        if results.detections:
            for id, detections in enumerate(results.detections):
                bboxC = detections.location_data.relative_bounding_box
                bbox = int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h)
                cv2.rectangle(img, bbox, (255,0,255),2)

class frames():
    def __init__(self):
        self.q = 0
        self.qn = 0
        self.cTime = time.time()
        self.pTime = time.time()
        self.fps = 0
    def frame_compute(self):
        self.cTime = time.time()
        self.q = (self.cTime-self.pTime)
        self.qn = self.q*.1 + self.qn*.9
        self.fps = 1/(self.qn)
        self.pTime = self.cTime
        return self.fps
    
    def frame_place(self, loc, img):
        cv2.putText(img, str(int(fps.frame_compute())), loc, cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 5)
        
# ### the code below will read off the video from a webcam
web_cam_n = 1
cap = cv2.VideoCapture(web_cam_n) ## the number defines the web_cam_number
detector = faceDetector()
fps = frames()
success, img = cap.read()
h,w,c = img.shape

while 1:
    success, img = cap.read()
    detector.findFaces(img,w,h)
    fps.frame_place((10,70), img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

