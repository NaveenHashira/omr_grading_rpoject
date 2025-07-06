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
cv2.imshow("region", region)
height, width = region.shape[:2]

crop_x = int(0.03 * width)
crop_y = int(0.065 * height)
crop_w = int(0.97 * width)
crop_h = int(0.95 * height)

cropped = region[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
cv2.imshow("Cropped Region", cropped)

questions_no_q_num = utils.split_and_crop_omr_bubbles(cropped, q_num_crop_ratio=0.17)

if questions_no_q_num is None or not questions_no_q_num:
    print("Error: Could not segment questions.")
    exit()

myPixelVal = np.zeros((50, 4))
question_index = 0

for row_of_questions in questions_no_q_num:
    for question_segment in row_of_questions:
        option_bubbles = utils.split_question_into_options(question_segment)

        if not option_bubbles or len(option_bubbles) != 4:
            print(f"Warning: Question {question_index + 1} did not yield 4 option bubbles. Skipping.")
            question_index += 1
            continue

        for option_idx, bubble_image in enumerate(option_bubbles):
            if bubble_image is not None and bubble_image.size > 0:
                total_pixels = cv2.countNonZero(bubble_image)
                myPixelVal[question_index][option_idx] = total_pixels
            else:
                myPixelVal[question_index][option_idx] = 0

        question_index += 1

print("Calculated pixel values for each option:")
print(myPixelVal)

option_map = ['A', 'B', 'C', 'D']
MIN_MARK_FILL_THRESHOLD = 70

with open("omr_answers.txt", "w") as f:
    for q_num_0_indexed in range(len(myPixelVal)):
        current_q_pixel_counts = myPixelVal[q_num_0_indexed]
        ans_index = np.argmin(current_q_pixel_counts)
        max_pixel_count = current_q_pixel_counts[ans_index]
        marked_options_indices = np.where(current_q_pixel_counts > MIN_MARK_FILL_THRESHOLD)[0]

        if len(marked_options_indices) == 1:
            answer = option_map[marked_options_indices[0]]
        elif len(marked_options_indices) > 1:
            answer = "MULTIPLE_MARKS"
            print(f"Warning: Q{q_num_0_indexed + 1} has multiple marks. Pixel counts: {current_q_pixel_counts}")
        else:
            answer = "UNANSWERED"
            print(f"Warning: Q{q_num_0_indexed + 1} is unanswered. Pixel counts: {current_q_pixel_counts}")

        f.write(f"Q{q_num_0_indexed + 1}: {answer}\n")

print("\nAnswers written to omr_answers.txt")
