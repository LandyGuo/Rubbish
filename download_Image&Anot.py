#coding=utf-8
import codecs
import re
import numpy as np
import urllib2
from PIL import Image,ImageDraw,ImageFont
from matplotlib import pyplot as plt



"""
1,http://insopenapitest.cn-hangzhou.oss.aliyun-inc.com/07502108022016030774/e7cb96e55fe93cf38c336a8081662a75_00022-ED025C6C-F926-4d12-91A4-CB0C1BBC6F1D.JPG?OSSAccessKeyId=Rex1c2QQNnDrmtzd&Signature=Ar2J0k%2F3EBl42zfmsKiebT9yn%2FM%3D&Expires=4620896280,"3(10.0,10.0,20.0,20.0,10)	(548.86,221.18,671.74,393.22,11)	(529.0,472.0,959.0,555.0,812)",正确#|#

"""

#download image

labels = set()
def extract_name(url):
    pattern = re.compile(r"/(.+\.(JPG|jpg)?)")
    m = re.search(pattern,url)
    name = m.group(1).split('/')[-1].rstrip(".JPG").rstrip(".jpg")
    return name


def download(url):
    binary_data = urllib2.urlopen(url)
    image = Image.open(binary_data)
    return image

def parselocations(locations,W,H):
    locs = []
    pattern = r"(.+)"
    regex_pattern = re.compile(pattern)
    for loc in locations:
        l = loc.strip()
        m = regex_pattern.match(l)
        if l and m:
            loc_str = m.groups(1)[0]
            loc_str = loc_str.lstrip("(").rstrip(")")
#             print loc_str
            coords =  loc_str.split(',')
            cls = coords[-1]
            labels.add(cls)
            x1,y1,x2,y2 = [float(x) for x in coords[:-1]]
            #normalize:center and normalize
            width,height = x2-x1,y2-y1
            center_x,center_y = x1+width/2,y1+height/2
            content = " ".join(map(str,[center_x*1.0/W,center_y*1.0/H,width*1.0/W,height*1.0/H]))+'\n'
            content = cls+" "+content
            locs.append(content)
    return locs

def saveImg2Dir(image,imagename):
    image.save("images/"+imagename+".jpg")

def saveText2Dir(lst,imagename):
    codecs.open("labels1/"+imagename+".txt",'w+','utf-8').writelines(lst)
    

LIST_MAP = {"341":15,"342":16}
Filter_TAG = set(["341","342"])
counter  ={}


def decode(line):
    lst = line.split(',')
    case_id = lst[0]
    url = lst[1]
#     print url

    image = download(url)
    name = extract_name(url)
    w,h =  image.size
    if not image:
        print "fail to download:",url
        return 
    num_pattern = re.compile(",\"(\d+)\(")
    loc_pattern = re.compile("\((.*?)\)")
    m = num_pattern.search(line)
    if not m:
        print "no number pattern:url:",url
        return 
    locations = []
    num =  int(m.group(1))
#     print num
    m2 = loc_pattern.findall(line)
    assert len(m2)==num,"Error:location number wrong:%s"%url
    for i in range(len(m2)):
        lst = m2[i].split(',')
        x1,y1,x2,y2= [float(x) for x in lst[:-1]]
        cls = lst[-1].strip()
        if cls not in Filter_TAG:
            continue
        if cls not in counter:
            counter[cls]=0
        counter[cls]+=1
        labels.add(cls)
#         print x1,y1,x2,y2,cls
        #normalize:center and normalize
        width,height = x2-x1,y2-y1
        center_x,center_y = x1+width/2,y1+height/2
        center_x,center_y,width,height= center_x*1.0/w,center_y*1.0/h,width*1.0/w,height*1.0/h
#         draw(image,center_x,center_y,width,height)
        content = " ".join(map(str,[center_x,center_y,width,height]))+'\n'
        content = str(LIST_MAP[cls])+" "+content
        locations.append(content)
    if locations:
        saveText2Dir(locations,name)
        saveImg2Dir(image,name)


cnt = 0
for line in codecs.open('new_added_kakou.txt','r','utf-8'):
    # if cnt<2817:
    #     cnt+=1
    #     continue
    if not line.strip():
    	continue
    print "current %s/4052"%cnt
    decode(line)
    cnt+=1

print "LABEL:"
labels = sorted(list(labels))
for label in labels:
    print label

for x,y in counter.items():
    print x,":",y
