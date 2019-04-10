#coding=utf-8
import requests
import threading
import os

url="http://rrc.cvc.uab.es/image/?ch=8&sample=%s&task=1"



def download_func(i):
	print("downloading:%s/%s"%(i, 9000))

	url_i = url % i
	bit_cnt = len(str(i))
	img_name = 'images/'+"img_"+'0'*(5-bit_cnt)+str(i)+'.jpg'
	if os.path.exists(img_name): return
	global sem
	sem.acquire()
	try:
		r = requests.get(url_i) 
		with open(img_name, "wb") as code:
			code.write(r.content)
	except:
		print("download failed:%s"%i)
	sem.release()


def download_thread(page_range):
	for x in page_range:
		download_func(x)

sem=threading.Semaphore(5)
threads = 10
task_ranges = [[] for _ in range(threads)]
for i in range(1, 9001):
	bin = i % threads
	task_ranges[bin].append(i)

thread_list = []
for i in range(threads):
	t = threading.Thread(target=download_thread, args=(task_ranges[i],))
	thread_list.append(t)

for t in thread_list:
	t.setDaemon(True)
	t.start()

for t in thread_list:
	t.join()
print("Done")




