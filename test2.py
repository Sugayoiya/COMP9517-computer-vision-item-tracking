import numpy as np 
import cv2, sys, time

cap = cv2.VideoCapture('1.mp4')
print('frame is {}/s'.format(cap.get(cv2.CAP_PROP_FPS)))

start = time.time()
count = 0
while(True):
    # capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # operations on the frame come here
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    count += 1
    
    # display the resulting frame
    cv2.imshow('frame',gray)
    cv2.waitKey(0)
    break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end = time.time()
print('total frams is {},est time is{}'.format( count,end - start ))
cap.release()
cv2.destroyAllWindows()