import numpy as np
import cv2

def Read(num, dir):
	img = []
	for k in range(num):
		filename = dir + "/%02d.png" % k
		img.append(cv2.imread(filename, cv2.IMREAD_UNCHANGED))
	print("[Done] Reading all the source images!")
	return img

def Write(num, dir, hdr):
	filename = dir + "/result.hdr"
	cv2.imwrite(filename, hdr[:,:,0:3])
	print("[Done] Writing the result image!")
