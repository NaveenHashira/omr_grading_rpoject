import cv2
import pytesseract
import re

# Path to the installed Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load and crop the image to the student name region
img = cv2.imread("OMR_ANSWER_KEY_page-0001.jpg")
name_box = img[222:262, 855:999]

# Preprocessing for better OCR
gray = cv2.cvtColor(name_box, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# OCR with line-level configuration
text = pytesseract.image_to_string(thresh, config="--psm 7").strip()

# Extract alphabetic part (student name)
match = re.search(r"[A-Za-z]+", text)
name = match.group(0) if match else text

print("Detected student name:", name)
