import cv2,sys

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if __name__ == '__main__':

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[0]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    # read video
    video = cv2.VideoCapture('1.mp4')
    print('read file')

    # exit if video not opened
    if not video.isOpened():
        print('can not open video')
        sys.exit()

    # read first frame
    ok, frame = video.read()
    if not ok:
        print('can not read video')
        sys.exit()
    
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # box = (x, y, width, height)
    box = (211,223,66,87)

    # box = cv2.selectROI(frame,False)
    print('box',box)

    ok = tracker.init(frame,box)

    while True:
        # read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timer = cv2.getTickCount()

        # update tracker
        ok, box = tracker.update(frame)

        # calculate frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # draw bounding box
        if ok:
            # tracking success
            p1 = (int(box[0]),int(box[1]))
            p2 = (int(box[0]+box[2]),int(box[1]+box[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0),2,1)
        else:
            cv2.putText(frame, "tracking failure detected",(100,80),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)

        cv2.putText(frame, tracker_type + "tracker",(100,20),cv2.FONT_ITALIC,0.75,(50,170,50),2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_ITALIC, 0.75, (50,170,50), 2)

        cv2.imshow('tracking',frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
