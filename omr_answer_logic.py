import cv2
import numpy as np
import utils
import easyocr
import matplotlib.pyplot as plt
import pytesseract

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
#cv2.imshow("region", region)
height, width = region.shape[:2]

crop_x = int(0.03 * width)
crop_y = int(0.065 * height)
crop_w = int(0.97 * width)
crop_h = int(0.95 * height)

cropped = region[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

cv2.imshow("cropped",cropped)

cv2.waitKey(0)


def organize_questions(image_array):
    if not isinstance(image_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    img_height, img_width = image_array.shape[:2]

    num_major_columns = 5
    num_major_rows_per_column = 10
    num_sub_columns_per_row = 4 # A, B, C, D

    row_height = img_height // num_major_rows_per_column
    
    major_col_width = img_width // num_major_columns

    single_bubble_width = major_col_width // num_sub_columns_per_row


    final_organized_list = []

    omr_question_counter = 1

    for major_col_idx in range(num_major_columns):
        for major_row_idx in range(num_major_rows_per_column):
            
            y_start_question_row = major_row_idx * row_height
            y_end_question_row = y_start_question_row + row_height

            # For each OMR question, we extract its 4 options (A, B, C, D)
            question_bubbles = [] 

            for sub_col_idx in range(num_sub_columns_per_row):
                # Calculate the X coordinates for the current bubble within the OMR question
                x_start_bubble = (major_col_idx * major_col_width) + (sub_col_idx * single_bubble_width)
                x_end_bubble = x_start_bubble + single_bubble_width

                y_end_question_row = min(y_end_question_row, img_height)
                x_end_bubble = min(x_end_bubble, img_width)

                
                bubble_cell = image_array[y_start_question_row:y_end_question_row, x_start_bubble:x_end_bubble]
                question_bubbles.append(bubble_cell)
            

            final_organized_list.append((omr_question_counter, question_bubbles))
            omr_question_counter += 1 
    
    return final_organized_list




all_cells = organize_questions(cropped)
myPixelVal = np.zeros((50, 4))

for q_idx, (omr_question_num, bubble_images_list) in enumerate(all_cells):
    
    for option_idx, image_cell in enumerate(bubble_images_list):    
        
        total_pixels = cv2.countNonZero(image_cell)
        myPixelVal[q_idx][option_idx] = total_pixels

print(myPixelVal)

option_map = ['A', 'B', 'C', 'D']

with open("omr_answers.txt", "w") as f:
    for i in range(len(myPixelVal)): 
        ans_index = np.argmin(myPixelVal[i])
        answer = option_map[ans_index]
        f.write(f"Q{i+1}: {answer}\n")
