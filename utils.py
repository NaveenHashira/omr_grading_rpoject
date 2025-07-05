import numpy as np
import cv2



def split_boxes_omr_id(img, rows=10, cols=10):
    boxes = []
    row_chunks = np.vsplit(img, rows)
    for row in row_chunks:
        col_chunks = np.hsplit(row, cols)
        for box in col_chunks:
            boxes.append(box)
    return boxes

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