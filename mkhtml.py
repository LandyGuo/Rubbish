#coding=utf-8

import os
import sys
import codecs


html_template_begin=r"""

<html> 
<head> 
<meta charset="utf-8" /> 
<title>display</title> 
<style> 

.row{
	border-bottom:5px solid #000000;
}

.image{
	margin:5px;
	display: inline-block;
	border:5px solid #ff0000;
}

.clear_float{
	clear: both;
}

.center{
	text-align: center;
}




</style> 
</head> 


<body> """


html_template_end = r"""
</body> 


</html> 

"""

display_row="""
	<div class="row"> 
		
		<div class="image"> <img src="predictions.png">123</img>      </div>
		<div class="image"> <img src="predictions.png"></img>      </div>

		<div class="image"> <img src="predictions.png"></img>      </div>
	</div>

"""
row_template  ="""
<div class="image"> <img src="{}"><div class="center">{}</div><div class="center">{}</div></img> </div>

"""


if __name__=="__main__":
	name_list = []
	print sys.argv
	for dirname in sys.argv[1:]:
		if not os.path.exists(dirname):
			print "dirname {} not exits!".format(dirname)
		filenames = os.listdir(dirname)
		name_list.append([])
		for filename in filenames:
			if filename.endswith("jpg") or filename.endswith("png"):
				name_list[-1].append(os.path.join(dirname,filename))

	if not name_list:
		print "Not provided image dir, can not display"
	length = len(name_list[0])
	for x in name_list[1:]:
		if len(x)!=length:
			print "Error:image nums are not equal !!"


	display  = list(zip(*name_list))


	# print display[0][0][0]

	# print display
	content = ""
	for row  in display:
		content+="""<div class="row">"""
		for image_path in row:
			# print "image_path: type",type(image_path),image_path
			content +=row_template.format(image_path,image_path.split('/')[-2],image_path.split('/')[-1])
		content+="""</div>"""

	codecs.open('index.html','w+','utf-8').write(html_template_begin+content+html_template_end)


