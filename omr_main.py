import cv2
import numpy as np
import utils  

path = "OMR_ANSWER_KEY_page-0001.jpg"
widthImg = 700
heightImg = 600

# Load image
img = cv2.imread(path)
print(img.shape)

# Extract name region
imgName = img[127:617, 639:1095]

# Extract and resize answer columns
imgAns_list = [
    img[704:1658, 145:280],
    img[704:1658, 350:486],
    img[704:1658, 555:690],
    img[704:1658, 761:898],
    img[704:1658, 967:1658]
]
imgAns_list = [cv2.resize(ans, (600, 600)) for ans in imgAns_list]

# Threshold all answer columns
thresholded_ans_list = []
for imgAns in imgAns_list:
    gray = cv2.cvtColor(imgAns, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
    thresholded_ans_list.append(thresh)

# Display first thresholded column
cv2.imshow("Thresholded_Ans_1", thresholded_ans_list[0])


all_cells = []  # will hold all 200 bubble images

for idx, thresh_img in enumerate(thresholded_ans_list):
    # split this column into 10 rows × 1 column × 4 bubbles
    cells = utils.split_boxes(thresh_img, num_rows=10, num_cols=1)
    print(f"Column {idx+1}: got {len(cells)} cells")  # should print 40
    all_cells.extend(cells)

print(f"Total bubbles extracted: {len(all_cells)}")  # should print 200

# Optional: display first few cells
for i, cell in enumerate(all_cells[:8]):
    cv2.imshow(f"bubble_{i+1:02d}", cell)

cv2.waitKey(0)
cv2.destroyAllWindows()
