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
omr_id_contour = utils.getNameContours(canny)
if omr_id_contour is not None:
    x, y, w, h = cv2.boundingRect(omr_id_contour)
    omr_id_region = canny[y:y+h, x:x+w]
    cv2.imshow("omr_id_region", omr_id_region)
else:
    print("OMR ID contour not found.")

print(omr_id_region.shape)
height, width = omr_id_region.shape[:2]

# Crop using proportional values (adjust if needed)
x = int(0.17 * width)
y = int(0.185 * height)
w = int(0.8 * width)
h = int(0.78 * height)

cropped_omr_id = omr_id_region[y:y+h, x:x+w]

cv2.imshow("Flexible Crop", cropped_omr_id)


cropped_omr_id = cv2.resize(cropped_omr_id, (200, 200))
cv2.imshow("resized", cropped_omr_id)
print("Shape before split:", cropped_omr_id.shape)
cells = utils.split_boxes_omr_id(cropped_omr_id,10,10)
print("no of cells: ",len(cells))  
for i, cell in enumerate(cells[:20]):
    cv2.imshow(f"Cell {i+1}", cell)

myPixelVal = np.zeros((10,10))
countC = 0
countR = 0

for image in cells:
    total_pixels = cv2.countNonZero(image)
    myPixelVal[countR][countC] = total_pixels
    countC+=1
    if countC == 10:
        countR+=1
        countC = 0
print(myPixelVal)

omr_id_digits = []

for col in range(10): 
    col_pixels = myPixelVal[:, col]  
    digit = np.argmin(col_pixels)    
    omr_id_digits.append(str(digit))

# Create ID string
omr_id_str = ''.join(omr_id_digits)
print("Detected OMR ID:", omr_id_str)





cv2.waitKey(0)