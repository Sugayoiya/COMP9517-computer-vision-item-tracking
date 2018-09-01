import numpy as np
import cv2
import sys,test
from collections import Counter

def get_kp_des(surf,frame,mask=None):
    if isinstance(frame,list):
        kp,des = [],[]
        for i in frame:
            kp_temp,des_temp = surf.detectAndCompute(i,mask)
            kp.append(kp_temp)
            des.append(des_temp)
    else:
        kp,des = surf.detectAndCompute(frame,mask)
    return kp,des

def draw_kp(frame,kp):
    if isinstance(frame,list):
        img = []
        for i in range(len(frame)):
            img_temp = cv2.drawKeypoints(frame[i],kp[i],None,(0,0,255),4)
            img.append(img_temp)
    else:
        img = cv2.drawKeypoints(frame,kp,None,(0,0,255),4)
    return img

def calculate_box(box,firstframe):
    boundbox =[firstframe[i[1]:i[1]+i[3],i[0]:i[0]+i[2]] for i in box] 
    return boundbox

def knnmatch(bf,des1,des2,k=2):
    if isinstance(des2,list):
        matches = []
        for i in des2:
            match_temp = bf.knnMatch(des1,i,k)
            matches.append(match_temp)
    else:
        matches = bf.knnMatch(des1,des2,k)
    return matches

def noisyleaveout(good):
    pass

if __name__ == '__main__':
    print('usage: python3 thisfile.py videofile box1 box2 box3 ...')
    print('example: python3 test8.py Video_sample_1.mp4 "537,50,40,175" "244,74,50,60" "353,312,215,32"')
    # print(len(sys.argv))

    # surf detector
    surf = cv2.xfeatures2d.SURF_create()

    # video file name
    videofile = sys.argv[1]
    # bound box defined
    box_size = len(sys.argv) - 2
    box = []
    for i in sys.argv[2:]:
        temp = i.split(',')
        temp = list(map(int,temp))
        box.append(temp)
    # print(videofile,box,len(box),box_size)

    # open video
    video = cv2.VideoCapture(videofile)

    if not video.isOpened():
        print('can not read video!')
        sys.exit()
    
    # read first frame
    ok, frame = video.read()
    if not ok:
        print('can not read video first frame!')
        sys.exit()
    # size of frame(RGB)
    size = frame.shape
    cv2.imshow('1st frame',frame)
    cv2.waitKey(0)

    # change to gray 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    boundbox_color = calculate_box(box,frame)
    boundbox = calculate_box(box,gray)
    # for i in boundbox:
    #     print(i.shape)
    #     cv2.imshow(''.format(i),i)
    #     cv2.waitKey(0)

    # boundbox keypoints and descriptors list type
    kp,des = get_kp_des(surf,boundbox)
    # keypoint img list type
    img = draw_kp(boundbox,kp)
    # for i in img:
    #     cv2.imshow('a',i)
    #     cv2.waitKey(0)

    while True:
        # read frame
        ok ,frame = video.read()
        if not ok:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # set timer
        timer = cv2.getTickCount()
        kp1, des1 = surf.detectAndCompute(gray,None)

        bf = cv2.BFMatcher()
        matches = knnmatch(bf,des1,des,k=2)
        print(type(matches))
        good = []
        for i in matches:
            temp = []
            for m,n in i:
                if m.distance < 0.45*n.distance:
                    temp.append([m])
            good.append(temp)

        # img3 = cv2.drawMatchesKnn(frame,kp1,boundbox[0],kp[0],good[0]\
        #             ,None,flags= 2)
        # cv2.imshow('img3',img3)
        # cv2.waitKey(0)      
        # img3 = cv2.drawMatchesKnn(img3,kp1,boundbox[1],kp[1],good[1]\
        #             ,None,flags= 2)
        # cv2.imshow('img3',img3)
        # cv2.waitKey(0)
        # img3 = cv2.drawMatchesKnn(img3,kp1,boundbox[2],kp[2],good[2]\
        #             ,None,flags= 2)
        # cv2.imshow('img3',img3)
        # cv2.waitKey(0)
        # break

        # draw keypoint matches on colored frame
        img3 = cv2.drawMatchesKnn(frame,kp1,boundbox_color[0],kp[0],good[0]\
                    ,None,flags= 2)      
        for i in range(1,len(boundbox)):
            img3 = cv2.drawMatchesKnn(img3,kp1,boundbox_color[i],kp[i],good[i]\
                    ,None,flags=2)
        cv2.imshow('img3',img3)
        cv2.waitKey(0)
        break

        # calculate frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)