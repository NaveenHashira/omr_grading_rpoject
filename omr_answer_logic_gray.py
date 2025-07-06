import cv2
import numpy as np
import utils
import easyocr
import matplotlib.pyplot as plt

path = "OMR_ANSWER_KEY_page-0001.jpg"

img = cv2.imread(path)
img = cv2.resize(img, (600, 600))
img_contours = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(gray, 100, 200)
ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

corners = utils.get_rectangle_corners(gray, utils.get_third_largest_contour, canny_img=canny)
region = utils.crop_rectangle_region(gray, corners)

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

# Convert BGR image to RGB for Matplotlib for the original image
img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

# Create a figure with multiple subplots to display all images
plt.figure(figsize=(15, 10)) # Adjust figure size as needed

# Plot original image
plt.subplot(2, 2, 1) # 2 rows, 2 columns, 1st plot
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')

# Plot grayscale image
plt.subplot(2, 2, 2) # 2 rows, 2 columns, 2nd plot
plt.imshow(gray, cmap='gray') # Grayscale images need cmap='gray'
plt.title('Grayscale Image')
plt.axis('off')

# Plot region image
plt.subplot(2, 2, 3) # 2 rows, 2 columns, 3rd plot
plt.imshow(region, cmap='gray')
plt.title('Region')
plt.axis('off')

# Plot cropped region image
plt.subplot(2, 2, 4) # 2 rows, 2 columns, 4th plot
plt.imshow(cropped, cmap='gray')
plt.title('Cropped Region')
plt.axis('off')

plt.tight_layout() # Adjust subplot params for a tight layout
plt.show()