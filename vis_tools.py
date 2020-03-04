# coding=utf-8

import cv2
import numpy as np
import os
import os.path as osp

from PIL import ImageFont
dirname = os.path.dirname(__file__)
Font = ImageFont.truetype(os.path.join(dirname, 'huawenfangsong.ttf'), 20)


def order_points(pts):
    pts = np.array(pts)
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    tl, bl = leftMost[np.argsort(leftMost[:, 1]), :]
    tr, br = rightMost[np.argsort(rightMost[:, 1]), :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]



class Vistool(object):

    def __init__(self, image_np_or_path):
        if isinstance(image_np_or_path, np.ndarray): 
            self._canvas = image_np_or_path
        else:
            self._canvas = cv2.imread(image_np_or_path)


    def save(self, savepath):
        if len(savepath.split('/'))>=2:
            savedir = osp.dirname(savepath)
            if not osp.exists(savedir):
                os.makedirs(savedir)
        cv2.imwrite(savepath, self._canvas)

    def draw(self, coords, color=[255, 0, 0], prefix='', canvas=None):
        if coords.shape[0]==0:
            return self._canvas
        if coords.shape[1]==8:
            coord = coords[:,:8]
            score = [1.0 for _ in range(coords.shape[0])]
        if coords.shape[1]==9:
            coord = coords[:,:8]
            score = coords[:, 8]
        if coords.shape[1]==5: #x1,y1,x2,y2,score
            score = coords[:, 4]
            x1, y1, x2, y2 = np.split(coords[:, :4], 4, axis=1)
            coord = np.concatenate([x1, y1, x2, y1, x2, y2, x1, y2], axis=1)

        coords = coord.reshape([-1, 4, 2])
        confs = score
        canvas = canvas if canvas else self._canvas
        self._canvas =  self.mold_vertext_on_image(canvas, coords, confs, color, prefix)

    def mold_vertext_on_image(self, image_np, coords, confs, color, prefix):
        """

        :param image_np:
        :param coords: batchx #vertext x 2
        :param content:
        :return:
        """
        num = coords.shape[0]
        if num==0: return image_np
        if coords.shape[1]==4:
            coords = np.array([order_points(coords[i]) for i in range(num)])
        vex_coords = np.reshape(coords, [num, -1, 2]).astype(np.int32)
        for i in range(num):
            cv2.polylines(image_np, [vex_coords[i]], 3, color=color, lineType=20)
        for i in range(num):
            loc = tuple(vex_coords[i][0])
            cv2.putText(image_np, prefix+' %.2f'%confs[i], loc,
                        cv2.FONT_ITALIC, 0.5, color, 1)
        return image_np


# Copied from https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/colormap.py
DETECTRON_PALETTE = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.167, 0.000, 0.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3) * 255


def draw(image_path, box_list, save_path, extra_info=[]):
    vt = Vistool(image_path)
    sources = list(set([getattr(box, 'source', '') for box in box_list]))
    colors = [hash(s)%DETECTRON_PALETTE.shape[0] for s in sources]
    for i,source in enumerate(sources):
        source_box_list = list(filter(lambda x: getattr(x, 'source', '')==source, box_list))
        coord = np.array([b.bndbox for b in source_box_list], dtype=np.float32) # nx8
        scores = np.array([b.label_score for b in source_box_list], dtype=np.float32) # n
        coords = np.concatenate([coord, np.expand_dims(scores, -1)], axis=1)
        vt.draw(coords, color=DETECTRON_PALETTE[colors[i]].astype(np.int32).tolist(), prefix=source )
    vt.save(save_path)
