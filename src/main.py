import cv2
import os
import Processing
import ML
import random
import sklearn
from sklearn import metrics
from sklearn import decomposition
from sklearn.externals import joblib

def main():
	current_dir =  os.path.abspath(os.path.dirname(__file__))
	parent_dir = os.path.abspath(current_dir + "/../")
	

	#img01 = cv2.imread(parent_dir + "/input/00746/00746_941201_fa.ppm")
	print("Loading Training and Validation sets...")
	TrainingSet, TrainingLabels, ValidationSet, ValidationLabels, num_classes = createSets(parent_dir)
	

	try:
		ml01 = joblib.load('svm01.pkl')
		ml02 = joblib.load('svm02.pkl')
		ml03 = joblib.load('svm03.pkl')
		ml04 = joblib.load('svm04.pkl')
		ml05 = joblib.load('svm05.pkl')
		ml06 = joblib.load('svm06.pkl')
		print("SVM's loaded")

	except Exception as e:
		ml01 = ML.MachineLearning((8,8))
		ml02 = ML.MachineLearning((12,12))
		ml03 = ML.MachineLearning((16,16))
		ml04 = ML.MachineLearning((20,20))
		ml05 = ML.MachineLearning((24,24))
		ml06 = ML.MachineLearning((28,28))

		
		# Training in multiple scales
		print ( "Training first classifier...")
		ml01.Train(TrainingSet, TrainingLabels)
		print ( "Training second classifier...")
		ml02.Train(TrainingSet, TrainingLabels)
		print ( "Training third classifier...")
		ml03.Train(TrainingSet, TrainingLabels)
		print ( "Training four classifier...")
		ml04.Train(TrainingSet, TrainingLabels)
		print ( "Training five classifier...")
		ml05.Train(TrainingSet, TrainingLabels)
		print ( "Training six classifier...")
		ml06.Train(TrainingSet, TrainingLabels)

		joblib.dump(ml01, 'svm01.pkl')
		joblib.dump(ml02, 'svm02.pkl')
		joblib.dump(ml03, 'svm03.pkl')
		joblib.dump(ml04, 'svm04.pkl')
		joblib.dump(ml05, 'svm05.pkl')
		joblib.dump(ml06, 'svm06.pkl')

		#print (ValidationLabels)
		#print("Ground Truth: " + str(ValidationLabels[0]))
		#pred = ml01.svm.predict(ValidationSet)
	
	pred = ml01.Validation(ValidationSet, ValidationLabels, 'svm')
	__test_metrics(ValidationLabels, pred)
	pred = ml02.Validation(ValidationSet, ValidationLabels, 'svm')
	__test_metrics(ValidationLabels, pred)
	pred = ml03.Validation(ValidationSet, ValidationLabels, 'svm')
	__test_metrics(ValidationLabels, pred)
	pred = ml04.Validation(ValidationSet, ValidationLabels, 'svm')
	__test_metrics(ValidationLabels, pred)
	pred = ml05.Validation(ValidationSet, ValidationLabels, 'svm')
	__test_metrics(ValidationLabels, pred)
	pred = ml06.Validation(ValidationSet, ValidationLabels, 'svm')
	__test_metrics(ValidationLabels, pred)


	'''
	resp = ml01.kNN(TrainingSet, TrainingLabels, ValidationSet[0], 10, kNN_type='cossine')
	print(resp, ValidationLabels[0])
	resp = ml02.kNN(TrainingSet, TrainingLabels, ValidationSet[0], 10, kNN_type='cossine')
	print(resp, ValidationLabels[0])
	resp = ml03.kNN(TrainingSet, TrainingLabels, ValidationSet[0], 10, kNN_type='cossine')
	print(resp, ValidationLabels[0])
	resp = ml04.kNN(TrainingSet, TrainingLabels, ValidationSet[0], 10, kNN_type='cossine')
	print(resp, ValidationLabels[0])
	resp = ml05.kNN(TrainingSet, TrainingLabels, ValidationSet[0], 10, kNN_type='cossine')
	print(resp, ValidationLabels[0])
	resp = ml06.kNN(TrainingSet, TrainingLabels, ValidationSet[0], 10, kNN_type='cossine')
	print(resp, ValidationLabels[0])
	'''
def createSets(parent_dir):


	Training = []
	Validation = []
	dir = parent_dir + "/input/"
	i = 0
	for dirpath, dirnames, filenames in os.walk(dir):
		if(len(filenames) != 1):
			num_train = int(len(filenames)*0.8)
			for index in range(len(filenames)):
				if(index < num_train):
					filename = filenames[index]
					if(filename.endswith('.ppm')):
						path = os.path.join(dir,filename[:5],filename)
						image = cv2.imread(path)
						Training.append([image, i])
				else:
					filename = filenames[index]
					if(filename.endswith('.ppm')):
						path = os.path.join(dir,filename[:5],filename)
						image = cv2.imread(path)
						Validation.append([image, i])
			i += 1

		
	random.shuffle(Training)
	random.shuffle(Validation)

	TrainSamples = []
	ValSamples = []
	TrainLabels = []
	ValLabels = []

	for sample in Training:
		ju_sample = sample[0]
		ju_label = sample[1]
		TrainSamples.append(ju_sample)
		TrainLabels.append(ju_label)

	for sample in Validation:
		bia_sample = sample[0]
		bia_label = sample[1]
		ValSamples.append(bia_sample)
		ValLabels.append(bia_label)

	return TrainSamples, TrainLabels, ValSamples, ValLabels, i

def __test_metrics(labels, predictions):
    print (metrics.accuracy_score(labels, predictions))
    print (metrics.classification_report(labels, predictions))
    print (metrics.confusion_matrix(labels, predictions).T)

if __name__ == '__main__':
	main()



