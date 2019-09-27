import numpy as np
import cv2
import matplotlib.pyplot as plt

global color
color = ["B", "G", "R"]

def ResponseCurve(num, img, exp):
	# arrange the basic information of variables
	sml = [cv2.resize(img[k], (10, 10)) for k in range(num)]
	l = 50				# constant that determines the importance of smoothness
	b = np.log2(exp)	# log2 shutter speed for images
	w = np.concatenate([np.arange(0, 128, 1), np.arange(127, -1, -1)])

	cur = []

	# consider R, G, B channels respectively
	for c in range(3):
		z = [np.array(sml)[k,:,:,c].flatten() for k in range(num)]
		imgNum = np.size(z, 0)	# num
		pixNum = np.size(z, 1)	# 10 * 10 = 100

		# according to formula proposed by debevec
		# X: 1 ~ n for g(n); n+1 ~ n+1+pixNum for ln(E)
		A = np.zeros((imgNum*pixNum + 256, 256 + pixNum))
		B = np.zeros((imgNum*pixNum + 256, 1))

		cnt = 0

		# arrange the information for 1st part of the formula (data-fitting)
		for i in range(imgNum):
			for p in range(pixNum):
				Z = z[i][p]
				# if (p == 4): print (Z)
				A[cnt][Z]     =  w[Z]	# w(z) * g(z)
				A[cnt][256+p] = -w[Z]	# w(z) * ln(E) * -1
				B[cnt]        =  w[Z] * b[i]
				cnt += 1

		# fix the curve by setting its middle value to 0
		A[cnt][128] = 1
		cnt += 1

		# arrange the information for 2nd part of the formula (smoothness)
		for n in range(256 - 1):
			A[cnt][n]   =      l * w[n+1]
			A[cnt][n+1] = -2 * l * w[n+1]
			A[cnt][n+2] =      l * w[n+1]
			cnt += 1

		# solve Ax = B through SVD
		x = np.linalg.lstsq(A, B, rcond = 0)[0]
		cur.append(x[:256])

		print("[Done] Computing response curve of %s channel!" % color[c])

	# plot out the figure
	plt.figure(figsize = (8, 6))
	plt.plot(cur[0], range(256), "bx")
	plt.plot(cur[1], range(256), "gx")
	plt.plot(cur[2], range(256), "rx")
	plt.ylabel("Pixel Value Z")
	plt.xlabel("Log Exposure X")
	plt.savefig("ResponseCurve.png")
	# plt.show()
	return cur

def RadianceMap(num, img, exp, cur):
	b = np.log2(exp)	# log shutter speed for images
	w = np.concatenate([np.arange(0, 128, 1), np.arange(128, 0, -1)])

	row = np.shape(img[0])[0]
	col = np.shape(img[0])[1]
	hdr = np.zeros((row, col, 3), "float32")

	# consider R, G, B channels respectively
	for c in range(3):
		z = [np.array(img)[k,:,:,c].flatten() for k in range(num)]
		
		imgNum = np.size(z, 0)	# num
		pixNum = np.size(z, 1)	# row * col

		# according to formula proposed by debevec
		E = np.zeros(pixNum)
		W = np.zeros(pixNum)

		# accumulate all the needed data for the formula
		for i in range(imgNum):
			for p in range(pixNum):
				Z = z[i][p]
				E[p] += w[Z] * (cur[c][Z] - b[i])
				W[p] += w[Z]

		W = np.where(W == 0, 1, W)
		E = E / W

		# update the final results of hdr
		hdr[:,:,c] = np.reshape(np.exp(E), (row, col))
		print("[Done] Constructing radiance map for %s channel!" % color[c])

	# plot out the figure
	plt.figure(figsize = (5, 6))
	plt.imshow(cv2.cvtColor(np.log2(hdr), cv2.COLOR_BGR2GRAY), cmap="jet")
	plt.colorbar()
	plt.savefig("RadianceMap.png")
	# plt.show()
	return hdr