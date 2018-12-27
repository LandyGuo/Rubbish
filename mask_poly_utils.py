#coding=utf-8
import numpy as np
import cv2

def format_mask2poly(mask, canvas=None):
    """
    :param mask: [0,255] mask， h x w
    :return:
    """
    ret, thresh = cv2.threshold(mask, 0, 255, 0) # thresh: [0,255]
    im2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    if canvas:
        cv2.drawContours(canvas, contours, -1, (255, 255, 255), 1)
        cv2.imwrite("contours.jpg", canvas)

    if contours:
        max_contour = contours[0]
        epsilon = 0.001*cv2.arcLength(max_contour, True)
        # convert to poly
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        if len(approx)==0: return ''
        # # vis
        # cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 3)
        poly = ' '.join(['%d,%d'%(i[0][0], i[0][1]) for i in approx])
    else:# no mask detected
        poly = ''
    return poly


def format_poly2mask(poly, msize, mask_value=255):
    """

    :param poly: str, formated poly str as "0,0 0,50 50,50 50,0"
    :param msize: tuple, output mask size, must be the original size
    :param mask_value: output mask with mask_value
    :return:
    """
    h, w = msize
    canvas_mask = np.zeros([h, w], dtype=np.uint8)
    poly_arr = np.array([list(map(int, x.split(','))) for x in poly.split()], dtype=np.int32)
    cv2.drawContours(canvas_mask, np.array([poly_arr]), -1, mask_value, -1)
    # cv2.imwrite('recover.jpg', canvas_mask)
    # print  "canvas_mask:", canvas_mask
    return canvas_mask.astype(np.uint8)


if __name__=='__main__':
    canvas = np.zeros([100,100], dtype=np.uint8)
    mask = canvas.copy()
    cv2.rectangle(mask, (0,0), (50,50), 255, -1)
    cv2.imwrite('origin.jpg', mask)

    # convert mask 2 poly
    poly = format_mask2poly(mask, canvas.copy())

    mask = format_poly2mask(poly, (100,100))
    print poly
