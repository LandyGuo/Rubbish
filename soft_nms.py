#coding=utf-8
import numpy as np

def soft_nms_cpu(
    boxes_in,
    iou_thr,
    method=1,
    sigma=0.5,
    min_score=0.001,
):
    boxes = boxes_in.copy()
    N = boxes.shape[0]
    inds = np.arange(N)

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]
        ti = inds[i]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]
        inds[i] = inds[maxpos]

        # swap ith box with position of max box
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = ts
        inds[maxpos] = ti

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below
        # threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > iou_thr:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov) / sigma)
                    else:  # original NMS
                        if ov > iou_thr:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight * boxes[pos, 4]

                    # if box score falls below threshold, discard the box by
                    # swapping with last box update N
                    if boxes[pos, 4] < min_score:
                        boxes[pos, 0] = boxes[N-1, 0]
                        boxes[pos, 1] = boxes[N-1, 1]
                        boxes[pos, 2] = boxes[N-1, 2]
                        boxes[pos, 3] = boxes[N-1, 3]
                        boxes[pos, 4] = boxes[N-1, 4]
                        inds[pos] = inds[N - 1]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    return boxes[:N], inds[:N]

if __name__=='__main__':
    # boxes and scores
    boxes = np.array([[200, 200, 400, 400], [220, 220, 420, 420], [200, 240, 400, 440], [240, 200, 440, 400], [1, 1, 2, 2]], dtype=np.float32)
    boxscores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    dets = np.concatenate([boxes, boxscores[:,None]], axis=1)
    print("dets:{}".format(dets))
    dets, inds = soft_nms_cpu(dets, 0.5)
    print("after softnms dets:{} inds:{}".format(dets, inds))





