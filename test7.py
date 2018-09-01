import numpy as np
import cv2
import sys,test
from collections import Counter

# sift = test.image_feature_detector(feat_type=0)
sift = cv2.xfeatures2d.SURF_create()

# def get_kp_des(frame,mask):
#     return sift.detector.detectAndCompute(frame, mask)

# def draw_kp(frame,kp):
#     return cv2.drawKeypoints(frame,kp,None,(0,0,255),4)

video = cv2.VideoCapture('Video_sample_1.mp4')

if not video.isOpened():
    print('can not open video!')
    sys.exit()

# read first frame
ok, frame = video.read()
if not ok:
    print('can not open video first frame')
    sys.exit()
# shape of frame
size = frame.shape
# cv2.imshow('1st frame',frame)
# cv2.waitKey(0)

# change to gray pic
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# bound box = (x, y, width, height)
# box = (205,230,83,80)
# box = (537,50,40,175)
box = (244,74,50,60)
# box = (353,312,215,32)
# face area [y1:y2, x1:x2]
face = frame[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]

kp2,des2 = sift.detectAndCompute(face,None)
img2 = cv2.drawKeypoints(face,kp2,None,(0,0,255),4)
x_center, y_center = 0,0
ptx_temp, pty_temp = 0,0 
while True:
    # read frame
    ok, frame = video.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    # set timer  
    timer = cv2.getTickCount()
    kp1, des1 = sift.detectAndCompute(gray, None)
    # img1 = cv2.drawKeypoints(gray,kp1,None,(0,0,255),4)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k = 2)
    good = []
    for m,n in matches:
        if m.distance < 0.45*n.distance:
            good.append([m])
    # print(good)
    img3 = cv2.drawMatchesKnn(frame,kp1,face,kp2,good,None,flags = 2)

    # calculate frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # get point(x,y) from keypoint
    if len(good)>0:
        blank = np.zeros(gray.shape)
        # blank = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
        no_noisy_good = []
        for i in good:
            (y1,x1) = kp1[i[0].queryIdx].pt
            (y2,x2) = kp2[i[0].trainIdx].pt
            # print(x1,y1,x2,y2)
            blank[int(x1),int(y1)] = 255
        cv2.imshow('kp',blank)
        # cv2.waitKey(0) 
        filtimg = cv2.boxFilter(blank,-1,(int(box[2]),int(box[3])))
        filtimg = cv2.boxFilter(filtimg,-1,(int(box[2]*2),int(box[3]*2)))
        # filtimg = cv2.boxFilter(blank,-1,(int(box[2]),int(box[3])))
        # filtimg = cv2.boxFilter(filtimg,-1,(int(box[2]),int(box[3])))
        cv2.imshow('filt',filtimg)
        min,max,min_loc,max_loc = cv2.minMaxLoc(filtimg)
        # print(max_loc)
        x_off, y_off = 0,0
        for i in good:
            (x1,y1) = kp1[i[0].queryIdx].pt
            # print(int(x1),int(y1))
            # print(max_loc)
            if abs(x1-max_loc[0])<= box[2] or abs(y1-max_loc[1])<= box[3]:
                no_noisy_good.append(i)

        lst = []
        # print(len(no_noisy_good))
        if len(no_noisy_good)>0:
            # print(no_noisy_good)
            for i in no_noisy_good:
                # tracking
                img1_idx = i[0].queryIdx
                img2_idx = i[0].trainIdx
                # print(good,'\n',good[0][0].queryIdx,good[0][0].trainIdx)
                # break
                (x1,y1) = kp1[img1_idx].pt
                (x2,y2) = kp2[img2_idx].pt
                # print((x1,y1),(x2,y2))

                # rectangle position
                pt1 = (int(x1)-int(x2),int(y1)-int(y2))
                pt2 = (int(pt1[0]+box[2]),int(pt1[1]+box[3]))
                lst.append((pt1,pt2))
                # print('pt1:',pt1,'pt2:',pt2)
            # print(Counter(lst))
            pt1,pt2 = Counter(lst).most_common(1)[0][0]
            # print(max(lst,key=lst.count))
            # pt1,pt2 =max(lst,key=lst.count)
            # print(pt1,pt2)

            # x_temp = int((pt2[0]+pt1[0])/2)
            # y_temp = int((pt2[1]+pt1[1])/2)
            # some smooth method
            # if ptx_temp == 0 and pty_temp == 0:
            #     ptx_temp = pt1[0]
            #     pty_temp = pt1[1]
            # elif abs(ptx_temp-pt1[0])>=box[2] or abs(pty_temp-pt1[1])>=box[2]:
            #     pt1 = (ptx_temp,pty_temp)
            #     pt2 = (int(pt1[0]+box[2]),int(pt1[1]+box[3]))
            # else:
            #     ptx_temp = pt1[0]
            #     pty_temp = pt1[1]

            cv2.rectangle(img3,pt1,pt2,(0,255,0),2,1)
            # print(x_off,y_off)
        
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

