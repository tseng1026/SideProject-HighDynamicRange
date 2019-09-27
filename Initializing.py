import numpy as np
import cv2

# 000 for black, 255 for white
def Align(num, img, times):
	bit = []	# images in binary (0 if pixel is small than thr)
	msk = []	# masks of images  (0 if pixel is too closed to thr)
	for k in range(num):

		# arrange the information of images
		mid = np.median(img[k])
		bit.append(cv2.threshold(img[k], mid, 255, cv2.THRESH_BINARY)[1])
		msk.append(cv2.inRange(img[k], mid - 10, mid + 10))

		offset = [0, 0]		# the offset of images compared to the first image (img[0])
		if k == 0: continue
		for t in range(times, -1, -1):

			# shrink the images
			bitImg = cv2.resize(bit[0], (0, 0), fx = 1 / (2**t), fy = 1 / (2**t), interpolation = cv2.INTER_NEAREST)
			mskImg = cv2.resize(msk[0], (0, 0), fx = 1 / (2**t), fy = 1 / (2**t), interpolation = cv2.INTER_NEAREST)
			bitCmp = cv2.resize(bit[k], (0, 0), fx = 1 / (2**t), fy = 1 / (2**t), interpolation = cv2.INTER_NEAREST)
			mskCmp = cv2.resize(msk[k], (0, 0), fx = 1 / (2**t), fy = 1 / (2**t), interpolation = cv2.INTER_NEAREST)
			mskInv = cv2.bitwise_not(mskImg)

			row = bitImg.shape[0]
			col = bitImg.shape[1]

			# consider the possible offset, and choose the one with min error
			off = [0, 0]
			err = row * col * 3
			for i in [-1, 0, 1]:
				for j in [-1, 0, 1]:
					offTmp = np.float32([[1, 0, i + offset[0]*2], [0, 1, j + offset[1]*2]])
					bitTmp = cv2.warpAffine(bitCmp, offTmp, (col, row))

					xorTmp = cv2.bitwise_xor(bitImg, bitTmp, mskInv)
					errTmp = np.sum(xorTmp // 255)
					if err > errTmp:
						err = errTmp
						off = [i + offset[0]*2, j + offset[1]*2]

			# update the round results of offset
			offset = off

		# update the eventual images
		offset = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
		img[k] = cv2.warpAffine(img[k], offset, (img[k].shape[1], img[k].shape[0]))

	print("[Done] Aligning all the images!")
	return img