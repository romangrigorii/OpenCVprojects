import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm


class handDetector():
    def __init__(self, mode = False, maxHands=2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode = self.mode, max_num_hands = self.maxHands, min_detection_confidence = self.detectionCon, min_tracking_confidence = self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, w, h):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    cv2.putText(img, str(int(id)), (int(lm.x*w),int(lm.y*h)), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1) # this will put the id down
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

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
detector = handDetector()
fps = frames()
success, img = cap.read()
h,w,c = img.shape

while 1:
    success, img = cap.read()
    detector.findHands(img,w,h)
    fps.frame_place((10,70), img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

