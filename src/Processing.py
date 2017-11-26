import cv2
import numpy as np


# Short data structure
class PointsLandmark:
	def __init__(self, x, y):
		self.x = x
		self.y = y


# Currently this function is using SIFT keypoint detector as landmark detector
def landmarkDetection(img):
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	     
	sift = cv2.xfeatures2d.SIFT_create()
	kp = sift.detect(gray,None)
	     

	#cv2.drawKeypoints(gray,kp, img)
	landmarks = []
	for keyPoint in kp:
		landmarks.append(PointsLandmark(keyPoint.pt[0], keyPoint.pt[1]))

	#cv2.imwrite('sift_keypoints.jpg',img)
	landmarks.sort(key=lambda elem: elem.x)

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







	







	




