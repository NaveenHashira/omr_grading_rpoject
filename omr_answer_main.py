import cv2
import numpy as np
import utils # Assuming utils.py contains helper functions for contour detection and cropping

# --- Configuration ---
OMR_IMAGE_PATH = "OMR_ANSWER_KEY_page-0001.jpg"
IMAGE_RESIZE_DIM = (600, 600)

# OMR Grid Structure
NUM_MAJOR_COLUMNS = 5
NUM_MAJOR_ROWS_PER_COLUMN = 10
NUM_SUB_COLUMNS_PER_ROW = 4 

OUTPUT_ANSWER_FILENAME = "omr_answers.txt"
OPTION_MAP = ['A', 'B', 'C', 'D']


def organize_questions(image_array):
    """
    Divides the pre-cropped OMR bubble region into individual question options.

    Args:
        image_array (np.ndarray): The image segment containing only OMR bubbles,
                                  with question numbers already cropped out.

    Returns:
        list: A list of tuples, where each tuple is
              (OMR_question_number, [list_of_4_bubble_images_for_that_question]).
    """
    if not isinstance(image_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    img_height, img_width = image_array.shape[:2]

    # Calculate dimensions for splitting the grid
    row_height = img_height // NUM_MAJOR_ROWS_PER_COLUMN
    major_col_width = img_width // NUM_MAJOR_COLUMNS
    single_bubble_width = major_col_width // NUM_SUB_COLUMNS_PER_ROW

    final_organized_list = []
    omr_question_counter = 1

    for major_col_idx in range(NUM_MAJOR_COLUMNS):
        for major_row_idx in range(NUM_MAJOR_ROWS_PER_COLUMN):
            y_start_question_row = major_row_idx * row_height
            y_end_question_row = y_start_question_row + row_height

            question_bubbles = []

            for sub_col_idx in range(NUM_SUB_COLUMNS_PER_ROW):
                x_start_bubble = (major_col_idx * major_col_width) + (sub_col_idx * single_bubble_width)
                x_end_bubble = x_start_bubble + single_bubble_width

                # Ensure extracted cells are within image boundaries
                y_end_question_row = min(y_end_question_row, img_height)
                x_end_bubble = min(x_end_bubble, img_width)

                bubble_cell = image_array[y_start_question_row:y_end_question_row, x_start_bubble:x_end_bubble]
                question_bubbles.append(bubble_cell)

            final_organized_list.append((omr_question_counter, question_bubbles))
            omr_question_counter += 1

    return final_organized_list


def main():
    """
    Main function to load OMR image, process it, and extract answers.
    """
    # --- 1. Load and Preprocess Image ---
    img = cv2.imread(OMR_IMAGE_PATH)
    if img is None:
        print(f"Error: Could not load image from {OMR_IMAGE_PATH}")
        return

    img = cv2.resize(img, IMAGE_RESIZE_DIM)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 10, 25)
    # Binary inverse threshold for finding filled bubbles (if used)
    thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # --- 2. Find and Crop OMR Region ---
    corners = utils.get_rectangle_corners(thresh, utils.get_third_largest_contour, canny, False)
    if corners is None or len(corners) != 4:
        print("Error: Could not find four corners for the OMR region.")
        return

    region = utils.crop_rectangle_region(canny, corners) # Using Canny output for region

    if region is None or region.size == 0:
        print("Error: Cropped region is empty or invalid.")
        return

    # Further cropping to isolate only the OMR bubbles section
    height, width = region.shape[:2]
    crop_x = int(0.03 * width)
    crop_y = int(0.065 * height)
    crop_w = int(0.97 * width) - crop_x # Calculate width from start to end
    crop_h = int(0.95 * height) - crop_y # Calculate height from start to end
    
    # Ensure crop_w and crop_h are positive
    crop_w = max(1, crop_w)
    crop_h = max(1, crop_h)

    cropped_omr_bubbles = region[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

    cv2.imshow("Cropped OMR Bubbles", cropped_omr_bubbles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- 3. Organize Questions and Extract Pixel Values ---
    all_cells = organize_questions(cropped_omr_bubbles)

    # Matrix to store pixel counts: 50 questions x 4 options
    myPixelVal = np.zeros((50, NUM_SUB_COLUMNS_PER_ROW), dtype=float)

    for q_idx, (omr_question_num, bubble_images_list) in enumerate(all_cells):
        for option_idx, image_cell in enumerate(bubble_images_list):
            # Count non-zero pixels (white pixels from Canny edges)
            total_pixels = cv2.countNonZero(image_cell)
            myPixelVal[q_idx][option_idx] = total_pixels

    
    # --- 4. Determine Answers and Write to File ---
    with open(OUTPUT_ANSWER_FILENAME, "w") as f:
        for i in range(len(myPixelVal)):
            ans_index = np.argmax(myPixelVal[i]) 
            answer = OPTION_MAP[ans_index]
            f.write(f"Q{i+1}: {answer}\n")

    print(f"\nAnswers written to {OUTPUT_ANSWER_FILENAME}")


if __name__ == "__main__":
    main()