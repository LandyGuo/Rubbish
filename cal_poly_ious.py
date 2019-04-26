#coding=utf-8
import cv2
import numpy as np
import time


# boxes: N x k x 2
# boxes: M x k x 2,
def cal_iou(poly_arr1, poly_arr2):
    poly1, poly2 = np.array(poly_arr1, dtype=np.int32), np.array(poly_arr2, dtype=np.int32)
    N, M = poly1.shape[0], poly2.shape[0]
    poly_c = np.concatenate([poly1.reshape([-1,2]), poly2.reshape([-1,2])], axis=0)
    min_x, max_x = np.min(poly_c[:,::2]), np.max(poly_c[:,::2])
    min_y, max_y = np.min(poly_c[:,1::2]), np.max(poly_c[:,1::2])
    poly1[:,:,::2]=poly1[:,:,::2]-min_x
    poly1[:,:,1::2]=poly1[:,:,1::2]-min_y
    poly2[:,:,::2]=poly2[:,:,::2]-min_x
    poly2[:,:,1::2]=poly2[:,:,1::2]-min_y
    canvas_width, canvas_height = int(round(max_x-min_x)+1), int(round(max_y-min_y)+1)
    canvas = np.zeros([canvas_height, canvas_width], dtype=np.uint8)
    poly1_masks, poly2_masks = [], []
    for i in range(N):
        mask = cv2.fillPoly(canvas.copy(), [poly1[i]], 1)
        poly1_masks.append(mask.reshape((-1,)))
    for i in range(M):
        mask = cv2.fillPoly(canvas.copy(), [poly2[i]], 1)
        poly2_masks.append(mask.reshape((-1,)))
    poly1_masks, poly2_masks = np.array(poly1_masks, dtype=np.float32), np.array(poly2_masks, dtype=np.float32)
    inter = np.dot(poly1_masks, poly2_masks.T) # N*M
    area1, area2 = np.sum(poly1_masks, axis=1), np.sum(poly2_masks, axis=1) # M, N
    area1, area2 = np.tile(area1, [N, 1]), np.tile(area2, [M, 1])
    union = (area1 + area2-inter).astype(np.float32)
    return inter/union

if __name__=="__main__":
    image = np.zeros((100, 100), dtype='uint8')
    box1 = np.array([[0, 0], [49, 0], [49, 49], [0, 49]], dtype=np.int32)
    box2 = np.array([[0, 0], [49, 0], [49, 49], [99, 49], [99, 99], [0, 99]], dtype=np.int32)
    print(cal_iou([box1], [box2]))
