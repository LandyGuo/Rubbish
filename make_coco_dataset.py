# coding: utf-8


import datetime
import cv2
import os
import os.path as osp
import codecs
import json
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


# COCO formart: http://cocodataset.org/#format-data

class MakeCOCODataset(object):

    def __init__(self, annotation_file, images_absdir, **kwargs):
        self.dataset = {}
        self.dataset['info'] = {"year" : 2021,
                     "version" : 'v1.0',
                     "description" : kwargs.get('description', 'temporary dataset'),
                     "contributor" : 'tester',
                     "url" : 'None',
                     "date_created" : datetime.date.isoformat(datetime.date.today())}
        self.dataset['license'] = {'id': 1234567,
                                   'name': 'NoLicense',
                                   'url': 'stillNo'}

        self.annotation_file = annotation_file
        self.images_absdir = images_absdir

        # for category convert
        self._id2category = {}
        self._category2id = {}

        # image_id convert
        self._id2imageName = {}
        self._imageName2id = {}

    def set_categories(self, cates):
        # 0 always represent background
        self.dataset['categories'] = []
        for i,c in enumerate(cates):
            self.dataset['categories'] +=[{'id':i, 'name':c, 'supercategory':'None'}]
            self._id2category[i] = c
            self._category2id[c] = i

    def set_images(self, ipath_list):
        self.dataset['images'] = []
        for i, ip in enumerate(ipath_list):
            filename = os.path.basename(ip)
            image = cv2.imread(ip)
            h, w, c = image.shape
            self.dataset['images'] += [{"id" : i,
                                        "width" : w,
                                        "height" : h,
                                        "file_name" : filename,
                                        "license" : 'None',
                                        "flickr_url" : 'None',
                                        "coco_url" : ip,
                                        "date_captured": datetime.date.isoformat(datetime.date.today())}]
            self._id2imageName[i] = filename
            self._imageName2id[filename] = i

    def set_annotations(self, annos): # annos:[{image_id:int, category_id:int, segmentation:RLE or [polygon], area:float, bbox:[x,y,width,height]}]
        self.dataset['annotations'] = []
        for i,a in enumerate(annos):
            self.dataset['annotations'] +=[{"id" : i,
                                           "image_id" : a['image_id'],
                                           "category_id" : a['category_id'],
                                           "segmentation" : a['segmentation'],
                                           "area" : a['area'],
                                           "bbox" : a['bbox'],
                                           "iscrowd" : 0 }]


    def _get_cates_images_annotations(self):
        # return cates:list
        #        images: abs_path list
        #        annotations:
        images_list = []
        cates_list = []
        annotations = []
        for line in codecs.open(self.annotation_file, 'r', 'utf-8'):
            line = line.strip()
            if not line: continue
            imagename, annos = line.split('\t')
            # collect image
            images_list+=[os.path.join(self.images_absdir, imagename)]

            # get image_id

            for an in annos.split('\002'):
                t, c, l  = an.split('\001')
                bbox = [int(float(x)) for x in l.split(',')] # x1, y1, x2, y2
                area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                res = {'image':imagename, # to change to image_id
                       'category':t,
                       'bbox': [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]], # x1, y1, w, h
                       'area': area,

                       }
                annotations+=[res]
        return images_list, cates_list, annotations

    def build_dataset(self, images_list=None, cates_list=None, annotations=None):

        if  images_list is None:
            images_list, cates_list, annotations = self._get_cates_images_annotations()

        logging.info('building image and category index...')
        # build indexes for image and category
        self.set_images(images_list)
        # ['BG', 'class1', 'class2', 'class3']
        self.set_categories(cates_list)

        def process_annotations(annotations):
            ret = []
            for i, a in enumerate(annotations):
                ret+=[{'image_id': self._imageName2id[a['image']],
                       'category_id': self._category2id[a['category']],
                       'area': a['area'],
                       'bbox': a['bbox'],
                       'segmentation': None
                     }]
            return ret

        logging.info('building annotation index...')
        annotations = process_annotations(annotations)
        self.set_annotations(annotations)


    def save_to_json(self, json_file):
        # Writing JSON data
        with open(json_file, 'w') as f:
            json.dump(self.dataset, f)


       
    
