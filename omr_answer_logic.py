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
cv2.imshow("region", region)
height, width = region.shape[:2]

crop_x = int(0.03 * width)
crop_y = int(0.05 * height)
crop_w = int(0.97 * width)
crop_h = int(0.95 * height)

crop_x = max(0, crop_x)
crop_y = max(0, crop_y)
crop_w = min(width - crop_x, crop_w)
crop_h = min(height - crop_y, crop_h)

cropped = region[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
cv2.imshow("Cropped Region", cropped)

num_rows_for_split = 4
num_cols_for_split = 5
bubbles_per_question_block = 4

current_cropped_h, current_cropped_w = cropped.shape[:2]

adjusted_cropped_h = (current_cropped_h // num_rows_for_split) * num_rows_for_split

# This is the line that needs modification
# The total horizontal split needed is num_cols_for_split * bubbles_per_question_block (which is 5 * 4 = 20)
# So, the adjusted width must be perfectly divisible by this total
adjusted_cropped_w = (current_cropped_w // (num_cols_for_split * bubbles_per_question_block)) * (num_cols_for_split * bubbles_per_question_block)

final_image_for_split = cropped[0:adjusted_cropped_h, 0:adjusted_cropped_w]

cv2.imshow("final_image_for_split", final_image_for_split)

all_cells = []
cells = utils.split_boxes_answer(final_image_for_split, num_rows_for_split, num_cols_for_split)
print("No of cells: ", len(cells))
all_cells.extend(cells)




#cv2.imshow("thresh",thresh)
#cv2.imshow("canny",canny)
cv2.waitKey(0)