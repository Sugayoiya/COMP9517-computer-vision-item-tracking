import numpy as np
import cv2
import sys
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
            img_temp = cv2.drawKeypoints(frame[i],kp[i],None,(0,0,255),2)
            img.append(img_temp)
    else:
        img = cv2.drawKeypoints(frame,kp,None,(0,0,255),2)
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

def getcolor(colors,i):
    temp = colors[i][0]
    temp = tuple(list(temp))
    temp = (int(temp[0]),int(temp[1]),int(temp[2]))
    return temp

# draw center in the box
def draw_cross(img, center, color, d):
    cv2.line(img,
             (center[0] - d, center[1] - d), (center[0] + d, center[1] + d),
             color, 3, cv2.LINE_AA, 0)
    cv2.line(img,
             (center[0] + d, center[1] - d), (center[0] - d, center[1] + d),
             color, 3, cv2.LINE_AA, 0)

# kalman filter
def kalman_filter(boundbox):
    x,y,w,h = boundbox[0],boundbox[1],boundbox[2],boundbox[3]
    # print(x,y,w,h)
    kalman = cv2.KalmanFilter(4,2,0)
    state = np.array([x+w/2,y+h/2,0,0],dtype='float64')
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                        [0., 1., 0., .1],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2,4)
    kalman.processNoiseCov = 1e-5 * np.eye(4,4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2,2)
    kalman.errorCovPost = 1e-1 * np.eye(4,4)
    kalman.statePost = state

    return kalman

# leave out some noisy point
def noisyleaveout(good,gray,kp1,kp2,box):
    if len(good)>0:
        blank = np.zeros(gray.shape)
        no_noisy_good = []
        for i in good:
            (y1,x1) = kp1[i[0].queryIdx].pt
            (y2,x2) = kp2[i[0].trainIdx].pt
            blank[int(x1),int(y1)] = 255
        # cv2.imshow('kp binary img',blank)
        # box filter
        # print(box)
        filtimg = cv2.boxFilter(blank,-1,(int(box[2]),int(box[3])))
        filtimg = cv2.boxFilter(filtimg,-1,(int(box[2]*2),int(box[3]*2)))
        # cv2.imshow('filted img',filtimg)
        # find the highest density kp in the img 
        # (the value is the biggest at that point after filting)
        min,max,min_loc,max_loc = cv2.minMaxLoc(filtimg)
        for i in good:
            (x1,y1) = kp1[i[0].queryIdx].pt
            # remove not good enough keypoint in drawing bound box 
            # not drawkeypoint()
            if int(abs(x1-max_loc[0]))<= box[2] or int(abs(y1-max_loc[1]))<= box[3]:
                no_noisy_good.append(i)
        
        lst = []
        if len(no_noisy_good)>0:
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
            # colors = tuple(list(colors))
            # print(colors,type(colors),type((255,0,0)),type(colors[1]))
            # colors = (int(colors[0]),int(colors[1]),int(colors[2]))
            # cv2.rectangle(img,pt1,pt2,colors,2,1)
            return pt1,pt2
    else:
        # tracking failure
        cv2.putText(img3, "detected failure",(100,20),cv2.FONT_ITALIC,0.75,(50,170,50),2)
        return 0,0


if __name__ == '__main__':
    print('usage: python3 thisfile.py videofile box1 box2 box3 ...')
    print('example: python3 test10.py Video_sample_1.mp4 "537,50,40,172" "244,74,50,60" "353,310,215,40"')
    # print(len(sys.argv))

    # surf detector
    # surf = cv2.xfeatures2d.SURF_create(25,8,8,True)
    # surf = cv2.xfeatures2d.SURF_create(1,1,1,1,0)
    surf = cv2.xfeatures2d.SURF_create()
    # surf = cv2.xfeatures2d.SIFT_create(100)

    # video file name
    videofile = sys.argv[1]
    # bound box defined
    box_size = len(sys.argv) - 2
    box = []
    trajectory = []
    for i in sys.argv[2:]:
        temp = i.split(',')
        temp = list(map(int,temp))
        box.append(temp)
        x = temp[0]+temp[2]//2
        y = temp[1]+temp[3]//2
        # trajectory
        trajectory.append([(x,y)])
    # print(videofile,box,len(box),box_size)
    # print(trajectory)

    #kalman filter
    kalman = []
    for i in box:
        # print(i)
        kalman.append(kalman_filter(i))

    # generate different colors
    labels = np.arange(box_size)+1
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    colors = cv2.merge([label_hue, blank_ch, blank_ch])
    colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)
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
    # cv2.imshow('1st frame',frame)
    # cv2.waitKey(0)

    # change to gray 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray frame',gray)
    # cv2.waitKey(0)
    
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

    # show the first frame keypoint
    display = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    for i in range(len(img)):
        display[box[i][1]:box[i][1]+box[i][3],box[i][0]:box[i][0]+box[i][2]] = img[i]
    cv2.imshow('1',display)
    cv2.waitKey(0)

    # draw initial keypoint matches on colored frame
    kp1, des1 = surf.detectAndCompute(gray,None)
    bf = cv2.BFMatcher()
    matches = knnmatch(bf,des1,des,k=2)
    # print(type(matches))
    good = []
    for i in matches:
        temp = []
        for m,n in i:
            if m.distance < 0.45*n.distance:
                temp.append([m])
        good.append(temp)

    img3 = cv2.drawMatchesKnn(frame,kp1,boundbox_color[0],kp[0],good[0]\
                ,None,flags= 2,matchColor = getcolor(colors,0))      
    for i in range(1,len(boundbox)):
        img3 = cv2.drawMatchesKnn(img3,kp1,boundbox_color[i],kp[i],good[i]\
                ,None,flags= 2,matchColor = getcolor(colors,i))
    cv2.imshow('initial keypoint matches',img3)
    cv2.waitKey(0)

    while True:
        # read frame
        ok ,frame = video.read()
        if not ok:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # set timer
        timer = cv2.getTickCount()
        kp1, des1 = surf.detectAndCompute(gray,None)
        anykp = draw_kp(frame,kp1)
        cv2.imshow('any frame keypoint',anykp)

        bf = cv2.BFMatcher()
        matches = knnmatch(bf,des1,des,k=2)
        # print(type(matches))
        good = []
        for i in matches:
            temp = []
            for m,n in i:
                if m.distance < 0.45*n.distance:
                    temp.append([m])
            good.append(temp)

        # kalman prediction
        prediction = []
        for i in kalman:
            a = i.predict()
            # print(i.predict(),type(a))
            prediction.append(a)

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
                    ,None,flags= 2,matchColor = getcolor(colors,0))      
        for i in range(1,len(boundbox)):
            img3 = cv2.drawMatchesKnn(img3,kp1,boundbox_color[i],kp[i],good[i]\
                    ,None,flags= 2,matchColor = getcolor(colors,i))
        # cv2.imshow('img3',img3)
        # cv2.waitKey(0)

        # calculate frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # tracking (get next x,y)
        for i in range(len(boundbox)):
            x,y = noisyleaveout(good[i],gray,kp1,kp[i],box[i])
            if x!=0 and y!=0:
                # print(box,x,y,type(x),type(y))
                measurement = (x[0]+int(box[i][2]/2),x[1]+int(box[i][3]/2))
                posterior = kalman[i].correct(measurement)
                # draw_cross(img3,(np.int32(posterior[0]),np.int32(posterior[1])),getcolor(colors,i),3)
                draw_cross(img3,(np.int32(posterior[0]),np.int32(posterior[1])),(255,255,255),3)
                cv2.rectangle(img3,x,y,getcolor(colors,i),2,1)
                trajectory[i].append((np.int32(posterior[0]),np.int32(posterior[1])))
            
            # draw trajectory
            for p in trajectory:
                for j in range(len(p)-1):
                    cv2.line(img3,p[j],p[j+1],getcolor(colors,i))
        
        # show FPS
        cv2.putText(img3, "FPS:"+str(int(fps)),(100,50),cv2.FONT_ITALIC,0.75,(50,170,255),2)
        cv2.imshow('match',img3)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    video.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
