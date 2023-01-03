import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

dir = 'pictures/scawy.jpg'

# ### READING IN IMAGES

img = cv.imread(dir)
cv.imshow('scary', img)
cv.waitKey(0)

# ### READING IN VIDEOS

# capture = cv.VideoCapture('') # capture is an instance of the class 
# while 1:
#     isTrue, frame = capture.read()
#     cv.imshow('Video', frame)
#     if cv.waitKey(20) & 0xFF==ord('d'): # if d is pressed, the video loop is broken out of 
#         break

# capture.release() # this releases the capture pointer
# cv.destroyAllWindows()
# cv.waitKey(0)

# ### CRETING WINDOWS OF DIFFERENT COLORS

# blank = np.zeros((500,500,3), dtype = 'uint8')
# cv.imshow('Blank', blank)
# blank[:] = 0,255,0
# cv.imshow('Green',blank)
# blank[:] = 0,0,255
# blank[100:200,300:400] = 255,0,0
# cv.imshow('Blue',blank)
# cv.rectangle(blank,(0,0),(250,400),(0,255,0), thickness=10)
# cv.putText(blank, "Hey stupid", (270,270), cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),2)
# cv.imshow('Rect',blank)
# cv.waitKey(0)

# ### CONVERTING IMAGE TO GREYSCALE

# img = cv.imread(dir)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('scary', gray)
# cv.waitKey(0)

# ###  BLURING IMAGE

# img = cv.imread(dir)
# cv.imshow('original', img)
# blur = cv.GaussianBlur(img, (9,9), cv.BORDER_DEFAULT)
# cv.imshow('scary', blur)
# cv.waitKey(0)

# ### EDGE DETECTION ON BLURRED IMAGE

# img = cv.imread(dir)
# canny = cv.Canny(cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT), 120, 120)
# cv.imshow('Canny', canny)
# cv.waitKey(0)

# ### DILATING THE LINES

# img = cv.imread(dir)
# canny = cv.Canny(cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT), 120, 120)
# dilated = cv.dilate(canny, (10,10), iterations = 3) 
# cv.imshow('Canny', dilated)
# cv.waitKey(0)

# ### RESIZING THE IMAGE

# img = cv.imread(dir)
# img = cv.resize(img, (500,500))
# cv.imshow('original', img)
# cv.waitKey(0)

# ### TRANSLATING THE IMAGE
# img = cv.imread(dir)
# def translate(img, x, y):
#     transMat = np.float32([[1,0,x],[0,1,y]])
#     dimensions = (img.shape[1],img.shape[0])
#     return cv.warpAffine(img, transMat,dimensions)
# cv.imshow('translated', translate(img,100,100))
# cv.waitKey(0)

# ### ROTATING THE IMAGE
# img = cv.imread(dir)
# def rotate(img, angle, rotPoint = None):
#     (height,width) = img.shape[:2]
#     if rotPoint is None:
#         rotPoint = (width//2, height//2)
#     rotMat = cv.getRotationMatrix2D(rotPoint,angle,1.0)
#     dimensions = (width,height)
#     return cv.warpAffine(img, rotMat, dimensions)
# cv.imshow('rotated', rotate(img,45))
# cv.waitKey(0)

# ### FINDING THE HISTOGRAM OF GRAYSCALE IMAGE THAT HAS A MASK APPLIED TO IT

# img = cv.imread(dir)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# blank = np.zeros(img.shape[:2], dtype = 'uint8')
# circle = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 500, 255, -1)
# mask = cv.bitwise_and(gray,gray, mask = circle)
# gray_hist = cv.calcHist([gray], [0],mask , [256], [0,256])
# cv.imshow('masked', mask)
# plt.plot(gray_hist)
# plt.show()
