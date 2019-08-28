'''
Created on 24-Aug-2019

@author: pawan
'''

import os
import cv2
import detect
from PIL import Image
import imutils
filename = './video15.mp4'

cap = cv2.VideoCapture(filename)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    image=Image.fromarray(frame)
    image.save('img.jpg')
    frame = imutils.rotate(frame, 270)

    text=detect.license_plate('img.jpg')
    cv2.imshow('Frame(press q to stop)',frame)
    print(text)
    _=os.system('cls')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

os.remove('img.jpg')
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
