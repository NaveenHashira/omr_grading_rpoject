import numpy as np
import cv2



def split_boxes(img, num_rows=10, num_cols=5):
    """
    Splits the input image into smaller cells by first dividing it into `num_rows` rows
    and `num_cols` columns per row, then further splitting each cell horizontally into 4 parts.

    This is useful when each grid cell (bubble region) contains multiple smaller bubbles (e.g. A–D options),
    and we want to isolate each option individually.

    Parameters:
        img (numpy.ndarray): The binary or grayscale image of the answer region.
        num_rows (int): Number of horizontal rows in the answer grid.
        num_cols (int): Number of columns per row (i.e. number of questions per row block).

    Returns:
        list of numpy.ndarray: A flat list of cell images (each representing one option bubble).
                               The total length will be num_rows * num_cols * 4.
    """

    # 1) split into horizontal strips
    rows = np.vsplit(img, num_rows)

    # 2) for each strip, split into columns
    boxes = []
    for r in rows:
        cols = np.hsplit(r, num_cols)
        boxes.extend(cols)

    single = []
    for b in boxes:
        cols = np.hsplit(b,4)
        for c in cols:
            single.append(c)    

    return single

def rectContours(contours): 

    rectCont = []

    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*peri,True)
            #print("Corner points: ",len(approx))
            if len(approx) == 4:
                rectCont.append(i)
    
    rectCont = sorted(rectCont,key=cv2.contourArea,reverse=True)
    
    return rectCont

def getCornerPoints(cont):
    peri = cv2.arcLength(cont,True)
    approx = cv2.approxPolyDP(cont,0.02*peri,True)
    return approx

def getNameContours(img):
    contours,_ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    img_area = img.shape[0] * img.shape[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 0.115 * img_area:
            return cnt