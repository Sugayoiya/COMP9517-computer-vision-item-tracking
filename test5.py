import numpy as np
import cv2,imutils as im
import sys,test

# suft = test.image_feature_detector(feat_type=0)
suft = cv2.xfeatures2d.SIFT_create()

# def get_kp_des(frame,mask):
#     return suft.detector.detectAndCompute(frame, mask)

# def draw_kp(frame,kp):
#     return cv2.drawKeypoints(frame,kp,None,(0,0,255),4)

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

kp2,des2 = suft.detectAndCompute(face,None)
img2 = cv2.drawKeypoints(face,kp2,None,(0,0,255),4)

while True:
    # read frame
    ok, frame = video.read()
    if not ok:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    # set timer  
    timer = cv2.getTickCount()
    kp1, des1 = suft.detectAndCompute(frame, None)
    img1 = cv2.drawKeypoints(frame,kp1,None,(0,0,255),4)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k = 2)
    good = []
    for m,n in matches:
        if m.distance < 0.55*n.distance:
            good.append([m])
    print(good)
    img3 = cv2.drawMatchesKnn(frame,kp1,face,kp2,good,None,flags = 2)

    # calculate frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # get point(x,y) from keypoint
    if len(good)>0:
        # tracking
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
        
    else:
        # tracking failure
        cv2.putText(img3, "detected failure",(100,20),cv2.FONT_ITALIC,0.75,(50,170,50),2)
    
    cv2.putText(img3, "FPS:"+str(int(fps)),(100,50),cv2.FONT_ITALIC,0.75,(50,170,255),2)
    cv2.imshow('match2',img3)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
video.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

