# coding=utf-8
import pycocotools.mask as cocomask

import cv2
import numpy as np
import time

# draw a rectangle
image = np.zeros((50, 50, 1), dtype=np.uint8)

# rectangle 1, 2
rec1 = cv2.rectangle(image.copy(), (0, 0), (24, 24), 1, -1)
rec2 = cv2.rectangle(image.copy(), (0, 0), (49, 24), 1, -1)

cv2.imwrite('rec1.jpg', rec1*255)
cv2.imwrite('rec2.jpg', rec2*255)


#  mask saved in numpy
def cal_mask_np(rec1, rec2):
    # cal area
    rec1, rec2 = np.transpose(rec1, (2, 0, 1)), np.transpose(rec2, (2, 0, 1))
    shape1, shape2 = rec1.shape, rec2.shape
    rec1, rec2 = np.reshape(rec1, (-1, shape1[1]*shape1[2])), np.reshape(rec2, (-1, shape1[1]*shape1[2]))
    area1, area2 = np.sum(rec1, axis=1), np.sum(rec2, axis=1) # N

    # cal inter
    rec2 = np.transpose(rec2, (1, 0))
    # print 'rec1.shape:%s np.sum:%s', (rec1.shape, np.sum(rec1))
    # print 'rec2.shape:%s np.sum:%s', (rec2.shape, np.sum(rec2))
    inter = np.dot(rec1.astype(np.float32), rec2.astype(np.float32)) # 1x1
    # mask iou
    return inter[0][0]/(area1[0]+area2[0]-inter[0][0])


############# use coco API
# mask saved in rle
def coco_calmask_rle(rec1, rec2):
    # encode both dt and gt mask with rle
    dt = cocomask.encode(np.array(rec1, order='F'))[0]
    dt['counts'] = dt['counts'].decode('ascii')
    # print dt

    gt = cocomask.encode(np.array(rec2, order='F'))[0]
    gt['counts'] = gt['counts'].decode('ascii')
    # print gt

    o = cocomask.iou([dt], [gt], np.zeros((1,), dtype=np.bool))[0][0]
    return o.astype('float32')

################### use poly mask
# mask saved in poly, use this format online
def coco_calmask_poly(rec1, rec2):
    # 获取坐标(x,y), 不包含顶点
    poly1 = np.array([0., 0, 25., 0, 25, 25,  0, 25], dtype=np.float32)
    poly2 = np.array([0,  0, 50., 0, 50, 25,  0, 25], dtype=np.float32)
    h, w = 50, 50
    dt = cocomask.frPyObjects([poly1], h, w)
    # print dt
    gt = cocomask.frPyObjects([poly2], h, w)
    # print gt
    o2 = cocomask.iou(dt, gt, np.zeros((1,), dtype=np.bool))[0][0]
    return o2.astype('float32')


def timer(func, run_times, *args):
    start = time.time()
    for _ in range(run_times):
        func(*args)
    end = time.time()
    print func.__name__, ":", (end-start)*1./run_times


if __name__=='__main__':
    timer(cal_mask_np, 1000, rec1.copy(), rec2.copy())
    timer(coco_calmask_rle, 1000, rec1.copy(), rec2.copy())
    timer(coco_calmask_poly, 1000, rec1.copy(), rec2.copy())
