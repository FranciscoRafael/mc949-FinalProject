import cv2
import numpy as np


# Short data structure
class PointsLandmark:
	def __init__(self, x, y):
		self.x = x
		self.y = y


# Currently this function is using SIFT keypoint detector as landmark detector
def landmarkDetection(x, y):
	landmarks = []
	for i in range (len(x)):
		landmarks.append(PointsLandmark(x[i], y[i]))

	Final_FV = []
	for point in landmarks:
		cordX = round(point.x)
		cordY = round(point.y)
		fv = HOG(img[cordX-2:cordX+3,cordY-2:cordY+3,:], cell_size=(1,1), block_size=(1,1))
		Final_FV.append(fv)

	return Final_FV


def HOG(img, cell_size=(28, 28), block_size=(2, 2), nbins=9):
	# winSize is the size of the image cropped to an multiple of the cell size
	hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
	                                  img.shape[0] // cell_size[0] * cell_size[0]),
	                        _blockSize=(block_size[1] * cell_size[1],
	                                    block_size[0] * cell_size[0]),
	                        _blockStride=(cell_size[1], cell_size[0]),
	                        _cellSize=(cell_size[1], cell_size[0]),
	                        _nbins=nbins)

	h = hog.compute(img)
	#print (h.shape)
	return h







	







	




