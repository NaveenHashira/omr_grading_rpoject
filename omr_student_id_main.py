import cv2
import numpy as np
import utils  

def extract_omr_id_from_path(path):
    """
    Extracts a 10-digit OMR ID from the given image path.

    Parameters:
        path (str): File path to the OMR sheet image.

    Returns:
        str: Detected OMR ID as a 10-digit string.
    """
    try:
        # Load and resize image
        img = cv2.imread(path)
        if img is None:
            return "Image not found."

        img = cv2.resize(img, (600, 600))

        # Preprocessing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 10, 25)

        # Get the contour of the OMR ID region
        omr_id_contour = utils.getNameContours(canny)
        if omr_id_contour is None:
            return "OMR ID contour not found."

        x, y, w, h = cv2.boundingRect(omr_id_contour)
        omr_id_region = canny[y:y+h, x:x+w]

        # Crop to just the 10x10 bubble grid using proportion
        height, width = omr_id_region.shape[:2]
        crop_x = int(0.17 * width)
        crop_y = int(0.185 * height)
        crop_w = int(0.8 * width)
        crop_h = int(0.78 * height)
        cropped = omr_id_region[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

        # Resize to cleanly split into 10x10 grid
        cropped = cv2.resize(cropped, (200, 200))

        # Split into 10x10 cells
        cells = utils.split_boxes_omr_id(cropped, 10, 10)

        # Analyze pixel intensities column-wise
        pixel_values = np.zeros((10, 10))
        count = 0
        for row in range(10):
            for col in range(10):
                pixel_values[row][col] = cv2.countNonZero(cells[count])
                count += 1

        # For each digit (column), find the row with the darkest bubble
        omr_id_digits = []
        for col in range(10):
            digit = np.argmin(pixel_values[:, col])
            omr_id_digits.append(str(digit))

        return ''.join(omr_id_digits)

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    print(extract_omr_id_from_path("OMR_ANSWER_KEY_page-0001.jpg"))