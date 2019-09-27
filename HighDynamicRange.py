import sys
import numpy as np
import ImageFileIO
import Initializing
import Constructing
import matplotlib.pyplot as plt

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print ("[Usage] python script <num of img> <dir of img>")
		sys.exit(0)
	
	# initialize the images
	num = int(sys.argv[1])
	dir = sys.argv[2]
	img = ImageFileIO.Read(num, dir)
	# img = Initializing.Align(num, img, 5)

	# construct the response curve and plot the figure
	exp = [32, 16, 8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.007125, 0.00390625, 0.001953125, 0.00097656525]
	cur = Constructing.ResponseCurve(num, img, exp)
	hdr = Constructing.RadianceMap  (num, img, exp, cur)
	print (np.array(hdr)[:,:,0])

	# np.save("hdr", np.array(hdr))
	# hdr = np.load("hdr.npy")

	# create hdr resultant image (brighter = mantissa * 2 ^ exponent)
	brighter = np.max(hdr, axis = 2)
	mantissa = np.zeros_like(brighter)
	exponent = np.zeros_like(brighter)
	np.frexp(brighter, mantissa, exponent)
	rgbvalue = mantissa * 256.0 / brighter

	res = np.zeros((np.shape(hdr)[0], np.shape(hdr)[1], 4), dtype=np.uint8)
	res[:,:,0] = np.around(hdr[:,:,2] * rgbvalue)
	res[:,:,1] = np.around(hdr[:,:,1] * rgbvalue)
	res[:,:,2] = np.around(hdr[:,:,0] * rgbvalue)
	res[:,:,3] = np.around(exponent + 128)

	fin = np.zeros((np.shape(hdr)[0], np.shape(hdr)[1], 3))
	fin[:,:,0] = res[:,:,2] * np.power(2, exponent)
	fin[:,:,1] = res[:,:,1] * np.power(2, exponent)
	fin[:,:,2] = res[:,:,0] * np.power(2, exponent)

	ImageFileIO.Write(1, ".", fin)