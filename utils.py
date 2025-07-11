import numpy as np
import cv2

def split_boxes_answer(img, num_rows, num_cols):
    if img.shape[0] % num_rows != 0 or img.shape[1] % num_cols != 0:
        raise ValueError("Image dimensions must be perfectly divisible by num_rows and num_cols for the initial split.")

    rows = np.vsplit(img, num_rows)

    single_bubbles = []
    for r in rows:
        question_blocks = np.hsplit(r, num_cols)

        for block in question_blocks:
            if block.shape[1] % 4 != 0:
                raise ValueError(f"Individual question block width ({block.shape[1]}) is not divisible by 4 for splitting into bubbles.")
            
            bubbles_in_block = np.hsplit(block, 4)
            single_bubbles.extend(bubbles_in_block)

    return single_bubbles

def get_rectangle_corners(thresh_img, get_contour_fn, canny_img=None, show=False):
    """
    Extracts rectangle corners from 3 known contours in a thresholded image.

    Parameters:
        thresh_img (ndarray): Binary or thresholded image.
        get_contour_fn (function): Function to extract the nth largest contour.
        canny_img (ndarray): Optional image for cropping/visualization.
        show (bool): Whether to display the cropped contours.

    Returns:
        dict: Dictionary with 4 corner coordinates.
              {'top_left': (x, y), 'bottom_left': (x, y), 'top_right': (x, y), 'bottom_right': (x, y)}
    """

    contour_indices = [3, 4, 5]
    contour_names = ["top_left", "bottom_left", "top_right"]
    coords = {}

    for idx, name in zip(contour_indices, contour_names):
        cnt = get_contour_fn(thresh_img, idx)
        x, y, w, h = cv2.boundingRect(cnt)
        if name == "top_right":
            coords[name] = (x+w, y)
        else:
            coords[name] = (x, y)

        if show and canny_img is not None:
            region = canny_img[y:y+h, x:x+w]
            cv2.imshow(f"{name}_contour", region)

    # Calculate bottom_right assuming axis-aligned rectangle
    top_left = coords["top_left"]
    bottom_left = coords["bottom_left"]
    top_right = coords["top_right"]
    bottom_right = (top_right[0], bottom_left[1])

    coords["bottom_right"] = bottom_right

    if show:
        print("Rectangle corners:")
        for name, pt in coords.items():
            print(f"{name}: {pt}")
        if canny_img is not None:
            img_display = canny_img.copy()
            for name, pt in coords.items():
                cv2.circle(img_display, pt, 5, (0, 255, 0), -1)
                cv2.putText(img_display, name, (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.imshow("Corners", img_display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return coords

def crop_rectangle_region(image, corners):
    """
    Crops a rectangular region from the image using corner coordinates.
    Assumes axis-aligned rectangle.

    Parameters:
        image (ndarray): The source image.
        corners (dict): Dictionary with 4 corners - top_left, top_right, bottom_left, bottom_right.

    Returns:
        region (ndarray): Cropped image region.
    """
    x1, y1 = corners["top_left"]
    x2, y2 = corners["bottom_right"]

    # Make sure coordinates are in proper top-left and bottom-right order
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1, x2)
    y_max = max(y1, y2)

    cropped = image[y_min:y_max, x_min:x_max]
    return cropped



def split_boxes_omr_id(img, rows=10, cols=10):
    boxes = []
    row_chunks = np.vsplit(img, rows)
    for row in row_chunks:
        col_chunks = np.hsplit(row, cols)
        for box in col_chunks:
            boxes.append(box)
    return boxes

def getNameContours(img):
    contours,_ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    img_area = img.shape[0] * img.shape[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 0.115 * img_area:
            return cnt
        
def get_third_largest_contour(img,number):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < number:
        print(f"Only {len(contours)} contours found.")
        return None

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return sorted_contours[number]



def showTopContours(img, original_img, top_n=4):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    
    # Sort contours by area (descending)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i, cnt in enumerate(sorted_contours[:top_n]):
        area = cv2.contourArea(cnt)
        percent = (area / img_area) * 100
        print(f"Top {i+1} Contour: Area = {area:.2f} ({percent:.2f}% of image)")

        display_img = original_img.copy()
        cv2.drawContours(display_img, [cnt], -1, (0, 255, 0), 3)
        cv2.putText(display_img, f"Contour {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow(f"Top {i+1} Contour", display_img)
        cv2.waitKey(0)
        cv2.destroyWindow(f"Top {i+1} Contour")

def split_omr_bubble_grid(cropped_bubble_region_img, q_num_crop_ratio=0.15):
    if cropped_bubble_region_img is None or cropped_bubble_region_img.size == 0:
        return None

    img_height, img_width = cropped_bubble_region_img.shape[:2]

    num_rows = 10
    num_columns = 5

    row_height = img_height // num_rows
    col_width = img_width // num_columns

    cropped_question_segments = []

    for row_idx in range(num_rows):
        row_segments = []
        for col_idx in range(num_columns):
            y_start = row_idx * row_height
            y_end = y_start + row_height
            x_start = col_idx * col_width
            x_end = x_start + col_width

            y_end = min(y_end, img_height)
            x_end = min(x_end, img_width)

            raw_segment = cropped_bubble_region_img[y_start:y_end, x_start:x_end]

            if raw_segment.shape[1] == 0:
                cropped_segment = raw_segment
            else:
                crop_pixels = int(raw_segment.shape[1] * q_num_crop_ratio)
                cropped_segment = raw_segment[:, crop_pixels:]

            row_segments.append(cropped_segment)
        cropped_question_segments.append(row_segments)

    return cropped_question_segments

def split_question_into_options(question_segment_img):
    if question_segment_img is None or question_segment_img.size == 0:
        return []

    seg_height, seg_width = question_segment_img.shape[:2]
    num_options = 4 

    option_width = seg_width // num_options
    
    option_segments = []
    for i in range(num_options):
        x_start = i * option_width
        x_end = x_start + option_width

        x_end = min(x_end, seg_width)

        option_bubble = question_segment_img[:, x_start:x_end]
        option_segments.append(option_bubble)

    return option_segments

def split_question_into_options(question_segment_img):
    if question_segment_img is None or question_segment_img.size == 0:
        return []
    seg_height, seg_width = question_segment_img.shape[:2]
    num_options = 4
    option_width = seg_width // num_options
    option_segments = []
    for i in range(num_options):
        x_start = i * option_width
        x_end = x_start + option_width
        x_end = min(x_end, seg_width)
        option_bubble = question_segment_img[:, x_start:x_end]
        option_segments.append(option_bubble)
    return option_segments

def split_and_crop_omr_bubbles(cropped_bubble_region_img, q_num_crop_ratio=0.15):
    if cropped_bubble_region_img is None or cropped_bubble_region_img.size == 0:
        return None
    img_height, img_width = cropped_bubble_region_img.shape[:2]
    num_rows = 10
    num_columns = 5
    row_height = img_height // num_rows
    col_width = img_width // num_columns
    cropped_question_segments = []
    for row_idx in range(num_rows):
        row_segments = []
        for col_idx in range(num_columns):
            y_start = row_idx * row_height
            y_end = y_start + row_height
            x_start = col_idx * col_width
            x_end = x_start + col_width
            y_end = min(y_end, img_height)
            x_end = min(x_end, img_width)
            raw_segment = cropped_bubble_region_img[y_start:y_end, x_start:x_end]
            if raw_segment.shape[1] == 0:
                cropped_segment = raw_segment
            else:
                crop_pixels = int(raw_segment.shape[1] * q_num_crop_ratio)
                cropped_segment = raw_segment[:, crop_pixels:]
            row_segments.append(cropped_segment)
        cropped_question_segments.append(row_segments)
    return cropped_question_segments