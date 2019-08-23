'''
Created on 19-Aug-2019

@author: pawan
'''


import cv2
import numpy as np
import imutils
import pytesseract
import sys
import time
import pandas as pd

#load image
img=cv2.imread('car6.jpg')

#resizing image
img=imutils.resize(img,width=800)

#gray scaling
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('1. Gray scale img',gray_img)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray_img = cv2.bilateralFilter(gray_img, 11, 17, 17)
#cv2.imshow("2 - Bilateral Filter", gray_img)

#finding edges
edges=cv2.Canny(gray_img,170,200)
#cv2.imshow("4 - Canny Edges", edges)

# Find contours based on Edges
( cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
NumberPlateCnt = None #we currently have no Number plate contour

# loop over our contours to find the best possible approximate contour of number plate
count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour
            break


# Drawing the selected contour on the original image
mask = np.zeros(gray_img.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(img,img,mask=mask)
cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
cv2.imshow("Final_image",new_image)

# Configuration for tesseract
config = ('-l eng --oem 1 --psm 3')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
# Run tesseract OCR on image
text = pytesseract.image_to_string(new_image, config=config)

print(text)

cv2.waitKey(0) #Wait for user input before closing the images displayed
