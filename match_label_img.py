# coding=utf-8
import codecs
import re
import cv2
import os

from utils import create_xmlfile_online
from functools32 import lru_cache


memorized = lru_cache(maxsize=10000)


class PatternMatcher(object):
    def __init__(self, pattern, _type):
        self.pattern = re.compile(pattern)
        self.type = _type

    def matchedType(self, pattern_str):
        return self.type if self.pattern.match(pattern_str) else None


class PatternGroupMatcher(object):

    def __init__(self, *args):
        for arg in args:
            assert isinstance(arg, PatternMatcher), arg
        self.patterns = args

    def find_type(self, pattern_str):
        for p in self.patterns:
            t = p.matchedType(pattern_str)
            if t:
                return t
        return None


class TxtLable(object):

    def __init__(self, txt_path):
        self.path = txt_path
        self.type_mather = PatternGroupMatcher(
            PatternMatcher(
                '^\d{12,20}$',
                'id'),
            PatternMatcher(
                '^\d{2,6}/\d{2,6}(/\d{2,6})?$',
                'date'),
            PatternMatcher(
                '^\d{4}$',
                'bank_id'),
            PatternMatcher(
                '^[A-Za-z_.0-9\W]+$',
                'footnote'),
        )

    @memorized
    def load(self):
        lines = codecs.open(self.path, 'r', 'utf-8').readlines()
        assert len(lines) == 1, lines
        line = lines[0].strip()
        d = {}
        for kv_str in line.split():
            print('self._path', self.path)
            print('kv_str:', kv_str)
            k, v = kv_str.split(u'{')
            t = self.type_mather.find_type(k)
            if not t:
                print "%s illeage k:%s t:%s" % (self.path, k, t)
                exit(1)
            print(k.encode('utf-8'))
            d[t] = [v.split(',') + [k]]
        return d


class Image(object):

    def __init__(self, imgpath):
        self._meta = {'path': imgpath,
                      'image_name': os.path.basename(imgpath)}
        self._imgptr = None

    @property
    def meta(self):
        imgnp = self.imgNumpy
        h, w, c = imgnp.shape
        self._meta['height'] = h
        self._meta['width'] = w
        return self._meta

    @property
    @memorized
    def imgNumpy(self):
        if not self._imgptr:
            self._imgptr = cv2.imread(self._meta['path'])
        return self._imgptr


def mold_rectangle_on_image(image_np, type, coords, content=''):
    # draw box
    x1, y1, x2, y2 = map(int, map(float, coords))
    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image_np, type + ':' + content, (x1, y1 - 10),
                cv2.FONT_ITALIC, 1, (0, 255, 255), 2)
    return image_np


class MergeImageAndAnotations(object):

    def __init__(self, imageObj, txtObj):
        assert isinstance(imageObj, Image), imageObj
        assert isinstance(txtObj, TxtLable), txtObj
        self._imageObj = imageObj
        self._txtObj = txtObj

    def create_xml_annotations(self, xml_path):
        w_value, h_value = self._imageObj.meta['width'], self._imageObj.meta['height']
        d = self._txtObj.load()
        att_dict = {}
        image_name = self._imageObj.meta['image_name']
        create_xmlfile_online(
            w_value,
            h_value,
            d,
            att_dict,
            xml_path,
            image_name)

    def display(self):
        image_np = self._imageObj.imgNumpy

        d = self._txtObj.load()
        for k, v in d.items():
            v = v[0]  # 同一类型可能有多个标注框
            image_np = mold_rectangle_on_image(image_np, k, v[:4], v[-1])
        cv2.imwrite('test.jpg', image_np)


if __name__ == '__main__':
    for i in range(3, 4):
        txt_name = 'Labels/train_%d.txs' % i
        img_name = 'JPEGImages/train_%d.jpg' % i
        output_xml = 'rotate_xmls_full_train/train_%d.xml' % i
        txt = TxtLable(txt_name)
        image = Image(img_name)
        merge = MergeImageAndAnotations(image, txt)
        # merge.create_xml_annotations(output_xml)
        merge.display()
