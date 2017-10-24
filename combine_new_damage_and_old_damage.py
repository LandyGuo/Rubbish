#coding=utf-8
import codecs
import re
import urllib2
from PIL import Image
import os

#image_id -> [bbox1, bbox2,...]

labelmap = {"11":(0,"scratch1"), 
"12":(1,"scratch2"),
"21":(2, "deform1"),
"22":(3, "deform2"),
"23":(4, "deform3"),
"31":(5, "crack1"),
"32":(6, "crack2"),
"33":(7, "crack3"),
"41":(8, "lightbroken_1"), 
"42":(9, "lightbroken_2"),
"43":(10, "lightbroken_3"), 
"51":(11, "mirror_1"),
"52":(12, "mirror_2"),
"53":(13, "mirror_3"),
"6":(14, "window"),
"341":(15, "kakou_1"),
"342":(16, "kakou_2"),
}



class hisBbox(object):

	def __init__(self, x1, y1, x2, y2, img_w, img_h, label):
		self.cx = (x1+x2)/2./img_w
		self.cy = (y1+y2)/2./img_h
		self.w =  (x2-x1)/img_w
		self.h = (y2-y1)/img_h
		self.img_w = img_w
		self.img_h = img_h
		self.label = self.convert(label)
		self.left = x1/img_w
		self.top = y1/img_h
		self.right= x2/img_w
		self.bottom = y2/img_h
		self.square = self.w*self.h

	def convert(self, label):
		return str(labelmap[label.strip()][0])

	def __str__(self):
		return " ".join(map(str, [self.label, self.cx, self.cy, self.w, self.h]))

	def cal_iou(self, coord2):
		l = max(self.left, coord2.left)
		r = min(self.right, coord2.right)
		t = max(self.top,coord2.top)
		b = min(self.bottom, coord2.bottom)
		if r-l <= 0 or b-t<=0:
		    return 0. 
		joint = (r-l)*(b-t)
		denorm = self.square + coord2.square - joint + 1e-6
		return joint/denorm



def download(url, imgDir=None):
	if not imgDir:
		binary_data = urllib2.urlopen(url)
		image = Image.open(binary_data)
	else:
		img_name = extract_name(url)
		image = Image.open(os.path.join(imgDir,img_name+".jpg"))
	return image


def create_if_not_exists(path_name, isdir=True):
	if os.path.exists(path_name):
		return
	os.mkdir(path_name) if isdir else os.mknod(path_name)


def saveImg2Dir(image, dire, imagename):
	create_if_not_exists(dire)
	image.save(os.path.join(dire,imagename+".jpg"))

def saveText2Dir(content, dire, imagename):
	create_if_not_exists(dire)
	codecs.open(os.path.join(dire, imagename+".txt"),'w+','utf-8').write(content)



def parse(mboxstr, url, img_w, img_h):
	ret = []
	num_pattern = re.compile("\"(\d+)\s*\(")
	loc_pattern = re.compile("\((.*?)\)")
	m = num_pattern.search(mboxstr)
	if not m:
		print "no number pattern:url:",mboxstr.encode("utf-8")
		return ret
	locations = []
	num =  int(m.group(1))
	#     print num
	m2 = loc_pattern.findall(mboxstr)
	assert len(m2)==num,"Error:location number wrong:%s"%url
	for i in range(len(m2)):
		lst = m2[i].split(',')
		x1,y1,x2,y2= [float(x) for x in lst[:-1]]
		label = lst[-1].strip()
		if label not in labelmap:
			print "label:",label
			continue
		ret.append(hisBbox(x1,y1,x2,y2,img_w,img_h,label ))
	return ret


def extract_name(url):
	pattern = re.compile(r"/(.+\.(JPG|jpg)?)")
	m = re.search(pattern,url)
	name = m.group(1).split('/')[-1].rstrip(".JPG").rstrip(".jpg")
	return name


def load_history_labels(his_file):
	img2labels = {}
	with codecs.open(his_file, 'r', 'utf-8') as f:
		for i, line in enumerate(f):
			# if i>10:
			# 	break
			line = line.strip()
			index = line.find(",")
			url  = line[:index]
			imgObj = None
			try:
				imgObj  = download(url)
			except:
				print "download url :%s failed!" % url
				continue
			img_w, img_h =  imgObj.size
			img_id =  extract_name(url)
			bboxStr = line[index+1:-1]
			mboxes = parse(bboxStr, url, img_w, img_h)
			if img_id not in img2labels:
				img2labels[img_id] = [imgObj]
			img2labels[img_id].extend(mboxes)
	return img2labels


def load_kakou_labels(kakou_file):
	img2labels = {}
	with codecs.open(kakou_file, 'r', 'utf-8') as f:
		for i, line in enumerate(f):
			# if i>10:
			# 	break
			line = line.strip()
			# print line.encode('utf-8')
			index1 = line.find(",")
			index2 = line.find(",", index1+1)
			url = line[index1+1:index2]
			img_id =  extract_name(url)
			imgObj = None
			try:
				imgObj  = download(url)
			except:
				print "download url :%s failed!" % url
				continue
			img_w, img_h =  imgObj.size
			kakou_mboxes = parse(line, url,img_w, img_h)
			if not kakou_mboxes:
				continue
			if img_id not in img2labels:
				img2labels[img_id] = [imgObj]
			img2labels[img_id].extend(kakou_mboxes)
	return img2labels


def combine(img2kakoulabels, img2hislabels, imgDir, labelDir):
	
	for img in img2kakoulabels.keys():
		saveImg2Dir(img2kakoulabels[img][0], imgDir, img)
		if img in img2hislabels:
			histlabels = img2hislabels[img][1:]
			kakou_labels = img2kakoulabels[img][1:]
			remove_his_indexes = []
			for i,hisbbox in enumerate(histlabels):
				if hisbbox.label==5:
					for kakou_bbox in kakou_labels:
						if kakou_bbox.cal_iou(hisbbox)>0.5:
							remove_his_indexes.append(i)
			remove_his_indexes = set(remove_his_indexes)
			collect_boxes = []
			for i,hisbox in enumerate(histlabels):
				if i not in remove_his_indexes:
					collect_boxes.append(hisbox)
			collect_boxes.extend(kakou_labels)
			content = "\n".join(map(str, collect_boxes))
		else:
			content = "\n".join(map(str, img2kakoulabels[img][1:]))
		saveText2Dir(content, labelDir, img)


if __name__=="__main__":
	his_file = u"kakou_history_label.txt"
	kakou_file = u"new_added_kakou.txt"
	imgDir = u"/Users/guoqingpei/Desktop/kakou_images/JPEGImages"
	labelDir = u"/Users/guoqingpei/Desktop/kakou_images/labels"
	print "loading:load_history_labels... "
	img2hislabels = load_history_labels(his_file)
	print "loading:load_kakou_labels..."
	img2kakoulabels = load_kakou_labels(kakou_file)
	combine(img2kakoulabels, img2hislabels, imgDir, labelDir)



