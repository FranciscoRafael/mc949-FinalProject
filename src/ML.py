import cv2
import numpy as np
import Processing
import sklearn
from sklearn import decomposition
from sklearn import svm
from sklearn import linear_model
from sklearn import metrics
from sklearn import grid_search
from sklearn import multiclass
# parameters for grid search

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
class MachineLearning:

	def __init__(self, cell_size):
		self.cell_size = cell_size
		self.mean = []
		self.std = []
		self.pca = []

	def Train(self, images, labels):
		
		FVS = []
		for img in images:
			fv = Processing.HOG(img, self.cell_size)
			FVS.append(np.array(fv))

		
		FV_norm = self.Train_Normalization(FVS)

		print (len(FV_norm[0]))
		print("PCA")
		FV_norm = self.ReduceDimension(FV_norm, 'train')
		print(FV_norm.shape)
		print ("Training svm ...")
		self.SVM(FV_norm, np.array(labels), 'ovr')
		#print ("Training Neural Net ...")
		#self.ann_model = self.NeuralNet(FV_norm, np.array([1,0,0]), 2)

	def Validation(self, images, labels, model, num_classes=10, image_query=None):

		FVS = []
		for img in images:
			fv = Processing.HOG(img, self.cell_size)
			FVS.append(np.array(fv))

		
		
		#print (FV_norm)
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
		
		## Implementation using opencv ####
		''''
		svm = cv2.ml.SVM_create()
		svm.train(train, cv2.ml.ROW_SAMPLE, labels)
		return svm
		'''
		## --------------------------- ####

	# it is necessary to implement the euclidian distance
	def kNN(self, train_images, labels, test_image, num_classes, K=3, kNN_type='euclidian'):
		
		FVS = []
		for img in train_images:
			fv = Processing.HOG(img, self.cell_size)
			FVS.append(np.array(fv))

		
		if(self.mean == [] and self.std == []):
			FV_norm = self.Train_Normalization(FVS)
		else:
			FV_norm = self.Validation_Normalization(FVS)
		if(self.pca == []):
			FV_norm = self.ReduceDimension(FV_norm, 'train')
		else:
			FV_norm = self.ReduceDimension(FV_norm, 'test')

		fv_test = Processing.HOG(test_image, self.cell_size)
		fv_test = [np.array(fv_test)]
		FV_test = self.Validation_Normalization(fv_test)
		FV_test = self.ReduceDimension(FV_test, 'test')
		print (FV_test)

		if(kNN_type == 'cossine'):
			distances = []
			for i in range(len(FV_norm)):
				cossine = np.dot(FV_norm[i], FV_test[0])/(np.linalg.norm(FV_norm[i])*np.linalg.norm(FV_test[0]))
				distances.append([cossine, labels[i], i])

			distances.sort(key=lambda x: x[0], reverse=True)

		elif(kNN_type == 'euclidian'):
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

	# apply PCA and/or LDA -- not done!
	def ReduceDimension(self, Input, phase):

		### -- implementation using opencv ---- ##
		
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
		'''
		if(phase == 'train'):
			transformed = []
			X = np.squeeze(np.array(Input).astype(np.float32))
			mean, eigenvectors = cv2.PCACompute(X, None, maxComponents=100)
			# use only the first num_components principal components

			self.mean_pca = mean
			self.eigenvectors = eigenvectors

			for sample in X:
				compressed_sample = cv2.PCAProject(sample, mean[0], eigenvectors)
				transformed.append(compressed_sample)

		elif(phase == 'test'):
			transformed = []
			X = np.squeeze(np.array(Input).astype(np.float32))
			for sample in X:
				compressed_sample = cv2.PCAProject(sample, self.mean_pca[0], self.eigenvectors)
				transformed.append(compressed_sample)

		return np.array(transformed, dtype=np.float32)
		'''
		return input_transformed
	# Change
	def Train_Normalization(self, data):
		#print (data[0])
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








		





		

