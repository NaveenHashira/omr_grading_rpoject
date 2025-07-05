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

# detecting contours
#contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)
omr_id_contour = utils.getNameContours(canny)
if omr_id_contour is not None:
    x, y, w, h = cv2.boundingRect(omr_id_contour)
    omr_id_region = canny[y:y+h, x:x+w]
    cv2.imshow("omr_id_region", omr_id_region)
else:
    print("OMR ID contour not found.")


# find rectangles
#rectCon = utils.rectContours(contours)
#biggest_contour = rectCon[0]
#print(len(biggest_contour))


# output
cv2.imshow("gray",gray)
cv2.imshow("blur",blur)
cv2.imshow("canny",canny)
cv2.imshow("thresh",thresh)
#cv2.imshow("Contours Detected", img_contours)


cv2.waitKey(0)