# -*- coding:utf-8 -*-
"""
This is a Python version used to implement the Soft NMS algorithm.
Original Paper：Improving Object Detection With One Line of Code
"""
import numpy as np
import tensorflow as tf




def wrap_cpu_nms(boxes, prob, cls, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    # class wise nms
    clses = np.unique(cls)
    ids = np.arange(boxes.shape[0]).astype(np.int32)
    keeps = []
    for c in clses:
        c_ids = np.where(cls==c)[0]
        c_boxes = boxes[c_ids]
        c_prob = prob[c_ids]
        keep_ids = py_slow_cpu_softnms(c_boxes, c_prob, Nt, sigma, thresh, method)
        keeps.extend(c_ids[keep_ids])
    return np.array(keeps, dtype=np.int32)


def py_slow_cpu_softnms(dets, sc, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = sc
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 0], dets[pos:, 0])
        yy1 = np.maximum(dets[i, 1], dets[pos:, 1])
        xx2 = np.minimum(dets[i, 2], dets[pos:, 2])
        yy2 = np.minimum(dets[i, 3], dets[pos:, 3])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(np.int32)

    return keep




if __name__=="__main__":
    # boxes and scores
    boxes = np.array([[200, 200, 400, 400], [220, 220, 420, 420], [200, 240, 400, 440], [240, 200, 440, 400], [1, 1, 2, 2]], dtype=np.float32)
    boxscores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    cls = np.int32([1,1,1,1,2])

    # tf.image.non_max_suppression 中 boxes 是 [y1,x1,y2,x2] 排序的。
    with tf.Session() as sess:
        index = sess.run(tf.image.non_max_suppression(boxes=boxes, scores=boxscores, iou_threshold=0.5, max_output_size=5))
        print(index)
        index = wrap_cpu_nms(boxes, boxscores, cls)
        print(index)
        # index = py_slow_cpu_softnms(boxes, boxscores, method=2)
        # selected_boxes = sess.run(tf.gather(boxes, index))
        # print(selected_boxes)
