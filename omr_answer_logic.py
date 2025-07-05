import cv2
import numpy as np
import utils

path = "OMR_ANSWER_KEY_page-0001.jpg"

# load the image
img = cv2.imread(path)
img = cv2.resize(img, (600, 600))
img_contours = img.copy()

# preprocessing the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0) 
canny = cv2.Canny(blur, 10, 25)
thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)[1]

#answer_contour  = utils.showTopContours(canny,img,6)
#for i in range(4):
#   third_largest_contour = utils.get_third_largest_contour(thresh,i+3)
#    if third_largest_contour is not None:
#        cv2.drawContours(img_contours, [third_largest_contour], -1, (0, 255, 0), 2)
 #       cv2.imshow(f"{i+3}rd Largest Contour", img_contours)
corners = utils.get_rectangle_corners(thresh,utils.get_third_largest_contour,canny,False)
region = utils.crop_rectangle_region(canny, corners)
cv2.imshow("Cropped Region", region)



#cv2.imshow("thresh",thresh)
#cv2.imshow("canny",canny)
cv2.waitKey(0)