#coding=utf-8
import base64, urllib2, os, json
import random
import threading
import codecs


class TaskManager(object):
	ip_list = []

	def __init__(self, image_dir, limit=None):
		self.image_dir = image_dir
		self._limit = limit

	def create_task(self):
		tasks = []
		for image_name in os.listdir(self.image_dir):
			file_name = os.path.join(image_dir, image_name)
			host = random.randint(0, len(TaskManager.ip_list) - 1)
			task = (file_name, TaskManager.ip_list[host])
			tasks.append(task)
		if self._limit is not None:
			tasks = tasks[:self._limit]
		return tasks

	def create_split_tasks(self, splits=5):
		tasks = self._split_tasks(self.create_task(), splits)
		if self._limit is not None:
			tasks = tasks[:self._limit]
		return tasks

	def _split_tasks(self, tasks, splits):
		total_length = len(tasks)
		num_splits = min(total_length, splits)
		split_tasks = [[] for _ in range(num_splits)]
		for i in range(total_length):
			bin_num = i % num_splits
			split_tasks[bin_num].append(tasks[i])
		return split_tasks


class Saver(object):

	def __init__(self, res_dir):
		self._res_dir = res_dir
		if not os.path.exists(self._res_dir):
			os.makedirs(self._res_dir)

	def dump(self, ret):
		if self.query_exist(ret[0]):
			return
		image_name, result = ret
		save_path = os.path.join(self._res_dir, image_name+'.json')
		json.dump(result, codecs.open(save_path, 'w+', 'utf-8'))

	def query_exist(self, image_name):
		save_path = os.path.join(self._res_dir, image_name+'.json')
		if os.path.exists(save_path):
			return True
		return False


class RequestThread(threading.Thread):

	def __init__(self, saver, func, tasklist, interval=1, *args, **kwargs):
		self.saver = saver
		self.func = func
		self.tasklist = tasklist
		self.interval = interval
		super(RequestThread, self).__init__(*args, **kwargs)

	def run(self):
		for task in self.tasklist:
			# print("task:",task)
			ret = self.func(*task)
			self.saver.dump(ret)
			import time
			time.sleep(self.interval)


def get_xhb_classify_result(filename, ip, port=10000, service_name='test'):
	url = 'http://{}:{}/{}'.format(ip, port, service_name)
	image_name = os.path.basename(filename)
	file_content = open(filename).read()
	data = {"image": base64.b64encode(file_content), 'Image_name':image_name}
	file_content = json.dumps(data)
	request = urllib2.Request(url, file_content)
	response = urllib2.urlopen(request)
	result = response.read()
	return image_name, result


image_dir = 'test'
save_dir = 'res'
saver = Saver(save_dir)
tasks = TaskManager(image_dir).create_split_tasks()

threads = [RequestThread(saver, get_xhb_classify_result, subtask) for subtask in tasks]
for t in threads:
	t.start()
for t in threads:
	t.join()
