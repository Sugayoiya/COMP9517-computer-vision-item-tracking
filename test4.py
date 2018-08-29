import numpy as np
import cv2,imutils as im
import sys,test

suft = test.image_feature_detector(feat_type=0)

def get_kp_des(frame,mask):
    return suft.detector.detectAndCompute(frame, mask)

def draw_kp(frame,kp):
    return cv2.drawKeypoints(frame,kp,None,(0,0,255),4)

video = cv2.VideoCapture('1.mp4')

if not video.isOpened():
    print('can not open video!')
    sys.exit()

# read first frame
ok, frame = video.read()
if not ok:
    print('can not open video first frame')
    sys.exit()

# change to gray pic
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# bound box = (x, y, width, height)
box = (205,230,83,80)

# face area [y1:y2, x1:x2]
face = frame[230:310,205:288]

kp1, des1 = suft.detector.detectAndCompute(frame, None)
kp2, des2 = suft.detector.detectAndCompute(face, None)

img1 = cv2.drawKeypoints(frame,kp1,None,(0,0,255),4)
img2 = cv2.drawKeypoints(face,kp2,None,(0,0,255),4)

cv2.imshow('a',img1)
cv2.imshow('b',img2)

# bf = cv2.BFMatcher()
# matches = bf.match(des1,des2)
# img3 = np.zeros((1,1))
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,img3,flags = 2)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k = 2)

good = [] # add good enough match point
for m,n in matches:
    # print(m.distance,n.distance)
    if m.distance < 0.2*n.distance:
        good.append([m])

# img3 = np.zeros((1,1))
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags = 2)
# cv2.imshow('match',img3)

# print(good)
# print(good[0][0].distance, good[0][0].imgIdx, good[0][0].queryIdx, good[0][0].trainIdx)

# get point(x,y) from keypoint
img1_idx = good[0][0].queryIdx
img2_idx = good[0][0].trainIdx

(x1,y1) = kp1[img1_idx].pt
(x2,y2) = kp2[img2_idx].pt
# print((x1,y1),(x2,y2))

# rectangle position
pt1 = (int(x1-x2),int(y1-y2))
pt2 = (int(pt1[0]+box[2]),int(pt1[1]+box[3]))

# print('pt1:',pt1,'pt2:',pt2)
cv2.rectangle(img3,pt1,pt2,(0,255,0),2,1)
cv2.imshow('match2',img3)

video.release()
cv2.waitKey(0)
cv2.destroyAllWindows()