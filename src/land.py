from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import Processing
import main
import os
import random
import sklearn
from sklearn import metrics
from sklearn import decomposition
from sklearn.externals import joblib

TUNED_PARAMETERS = [
  {
    'estimator__kernel': ['rbf'],
    'estimator__gamma': [1e0, 1e-1, 1e-2, 1e-3, 1e-4],
    'estimator__C': [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
  }
]

parameters = {
    'penalty': ['l2'],
    'alpha': [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
}


SCORING = 'accuracy'
CV = 5
VERBOSE = 10
Cs = 10

# parameters for fitting svm

KERNEL = "linear"
GAMMA = 0.001
C = 0.0625
PROBABILITY = True

N_COMPONENTS = 512


def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def VectorLand_FV (image):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)
	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		(x, y, w, h) = face_utils.rect_to_bb(rect)

	FINALVECTOR = Processing.landmarkDetection(x, y)

	return FINALVECTOR


class MachineLearningLandMarks:

	def __init__(self, cell_size):
		self.cell_size = cell_size
		self.mean = []
		self.std = []
		self.pca = []

	def Train(self, images, labels):
		FVS = []
		for img in images:
			fv = VectorLand_FV(img)
			FVS.append(np.array(fv))
		FV_norm = self.Train_Normalization(FVS)
		FV_norm = self.ReduceDimension(FV_norm, 'train')
		self.SVM(FV_norm, np.array(labels), 'ovr')

	def Validation(self, images, labels, model, num_classes=10, image_query=None):
		FVS = []
		for img in images:
			fv = VectorLand_FV(img)
			FVS.append(np.array(fv))
		if(model == 'svm'):
			FV_norm = self.Validation_Normalization(FVS)
			FV_norm = self.ReduceDimension(FV_norm, 'test')
			res = self.svm.predict(FV_norm)
		elif(model == 'kNN'):
			res = self.kNN(images, labels, image_query, num_classes)
		return res


	def SVM(self,train, labels, multiclass_scheme):
		clf = svm.SVC()
		if (multiclass_scheme == 'ovr'):
			model = multiclass.OneVsRestClassifier(clf)
		if (multiclass_scheme == 'ovo'):
			model = multiclass.OneVsOneClassifier(clf)
		clf = grid_search.GridSearchCV(model, TUNED_PARAMETERS, scoring=SCORING, cv=CV, verbose=VERBOSE)
		clf.fit(train, labels)
		self.svm = clf

	def kNN(self, train_images, labels, test_image, num_classes, K=3, kNN_type='euclidian'):
		
		FVS = []
		for img in train_images:
			fv = VectorLand_FV(img)
			FVS.append(np.array(fv))

		if(self.mean == [] and self.std == []):
			FV_norm = self.Train_Normalization(FVS)
		else:
			FV_norm = self.Validation_Normalization(FVS)
		if(self.pca == []):
			FV_norm = self.ReduceDimension(FV_norm, 'train')
		else:
			FV_norm = self.ReduceDimension(FV_norm, 'test')

		fv_test = VectorLand_FV(test_image)
		fv_test = [np.array(fv_test)]
		FV_test = self.Validation_Normalization(fv_test)
		FV_test = self.ReduceDimension(FV_test, 'test')
		print (FV_test)

		if(kNN_type == 'euclidian'):
			distances = []
			for i in range(len(FV_norm)):
				euclidian = np.linalg.norm(FV_norm[i] - FV_test)
				distances.append([euclidian, labels[i], i])
			distances.sort(key=lambda x: x[0])
		else:
			print("Please, choose between 'euclidian' or 'cossine'")
			exit()
		votes = [0]*num_classes
		for j in range(K):
			label = distances[j][1]
			votes[label] += 1
		voted_label = votes.index(max(votes))
		return voted_label
	def ReduceDimension(self, Input, phase):	
		SizeToPCA = len(Input)//5
		if(phase == 'train'):
			print(1)
			pca = decomposition.TruncatedSVD(N_COMPONENTS)
			print(2)

			pca.fit(Input[:SizeToPCA])
			self.pca = pca
			input_transformed = pca.transform(Input)

		elif(phase == 'test'):
			input_transformed = self.pca.transform(Input)

		return input_transformed

	def Train_Normalization(self, data):
		mean = np.array([0.0]*len(data[0]))
		num_feat = len(data[0])
		std = [0.0]*len(data[0])

		
		for sample in data:
			for i in range(num_feat):
				mean[i] += sample[i][0]

		for i in range(num_feat):
			mean[i] = mean[i]/len(data)

		for sample in data:
			for i in range(num_feat):
				std[i] += (sample[i][0] - mean[i])**2

		for i in range(num_feat):
			std[i] = np.sqrt(std[i]/(len(data)-1))

		self.mean = mean
		self.std = std

		new_data = []

		for sample in data:
			new_sample = [0.0]*num_feat
			for i in range(num_feat):
				if(self.std[i] != 0):
					new_sample[i] = (sample[i][0] - mean[i])/std[i]
				else:
					new_sample[i] = 0
			new_data.append(np.float32(new_sample)) # Changed here to test Neural Net

		return np.array(new_data, np.float32)

	def Validation_Normalization(self, data):

		num_feat = len(data[0])
		new_data = []
		for sample in data:
			new_sample = [0.0]*num_feat
			for i in range(num_feat):
				if(self.std[i] != 0):
					new_sample[i] = (sample[i][0] - self.mean[i])/self.std[i]
				else:
					new_sample[i] = 0
			new_data.append(new_sample)

		return np.array(new_data, np.float32)



 print("Loading Training and Validation sets for LandMarks")
	TrainingSet, TrainingLabels, ValidationSet, ValidationLabels, num_classes = main.createSets(parent_dir)


try:
	svm1 = joblib.load('svmLANDMarks.pkl')
except Exception as e:

		print("SVM - LANDMARKS")
		svmTrain = MachineLearning((8,8))
		print ( "Training SVM classifier for Landmarks")
		svmTrain.Train(TrainingSet, TrainingLabels)

		joblib.dump(svmTrain, 'svmLANDMarks.pkl')

		pred = ml01.Validation(ValidationSet, ValidationLabels, 'svm')
		main.__test_metrics(ValidationLabels, pred)

		print("KNN - LANDKMARKS")

		knn = MachineLearning((8,8))
		print("Training kNN Classfier for Landmarks")
		resp = knn.kNN(TrainingSet, TrainingLabels, ValidationSet, 3, kNN_type = 'euclidian')
		print(resp)