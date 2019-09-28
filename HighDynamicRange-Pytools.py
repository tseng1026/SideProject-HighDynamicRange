import sys
import numpy as np
import cv2

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print ("[Usage] python script <num of img> <dir of img>")
		sys.exit(0)
	num = int(sys.argv[1])
	dir = sys.argv[2]

	img = []
	for k in range(num):
		filename = dir + "/%02d.png" % k
		img.append(cv2.imread(filename, cv2.IMREAD_UNCHANGED))

	lst = [32, 16, 8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.007125, 0.00390625, 0.001953125, 0.00097656525]
	exp = np.array(lst[0:num], dtype=np.float32)

	Align = cv2.createAlignMTB()
	Align.process(img, img)

	ResponseCurve = cv2.createCalibrateDebevec()
	cur = ResponseCurve.process(img, exp)

	RadianceMap = cv2.createMergeDebevec()
	hdr = RadianceMap.process(img, exp, cur)

	cv2.imwrite("result.hdr", hdr)