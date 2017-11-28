from sklearn.externals import joblib
from sklearn import metrics

def protocol_0(images, labels, images_test, labels_query, models):


	final_predictions = []
	


	pred01 = models[0].kNN(images, labels, images_test, 10)
	pred02 = models[1].kNN(images, labels, images_test, 10)
	pred03 = models[2].kNN(images, labels, images_test, 10)
	pred04 = models[3].kNN(images, labels, images_test, 10)
	pred05 = models[4].kNN(images, labels, images_test, 10)
	pred06 = models[5].kNN(images, labels, images_test, 10)

	__test_metrics(labels_query, pred01)
	__test_metrics(labels_query, pred02)
	__test_metrics(labels_query, pred03)
	__test_metrics(labels_query, pred04)
	__test_metrics(labels_query, pred05)
	__test_metrics(labels_query, pred06)




	
		
	for num in range(len(pred01)):
		votes = [0]*10
		votes[pred01[num]] += 1
		votes[pred02[num]] += 1
		votes[pred03[num]] += 1
		votes[pred04[num]] += 1
		votes[pred05[num]] += 1
		votes[pred06[num]] += 1

		label_voted = votes.index(max(votes))
		final_predictions.append(label_voted)
		

	return final_predictions
	#joblib.dump(final_predictions, 'protocol_0.pkl')
	#print(labels_query)

def protocol_1(images, labels, images_test, labels_query, models):

	final_predictions = []
	i = 0
	for query_image in images_test:
		votes = [0]*10
		print("C1", i)
		pred01 = models[0].kNN(images, labels, query_image, 10, kNN_type='cossine')
		print(pred01)
		print("C2", i)
		pred02 = models[1].kNN(images, labels, query_image, 10, kNN_type='cossine')
		print("C3", i)
		pred03 = models[2].kNN(images, labels, query_image, 10, kNN_type='cossine')
		print("C4", i)
		pred04 = models[3].kNN(images, labels, query_image, 10, kNN_type='cossine')
		print("C5", i)
		pred05 = models[4].kNN(images, labels, query_image, 10, kNN_type='cossine')
		print("C6", i)
		pred06 = models[5].kNN(images, labels, query_image, 10, kNN_type='cossine')
		i += 1

		votes[pred01] += 1
		votes[pred02] += 1
		votes[pred03] += 1
		votes[pred04] += 1
		votes[pred05] += 1
		votes[pred06] += 1

		label_voted = votes.index(max(votes))
		final_predictions.append(label_voted)

	print(final_predictions)
	joblib.dump(final_predictions, 'protocol_1.pkl')
	print(labels_query)

def protocol_2(ValidationSet, ValidationLabels, models):

	pred01 = models[0].Validation(ValidationSet, ValidationLabels, 'svm')
	pred02 = models[1].Validation(ValidationSet, ValidationLabels, 'svm')
	pred03 = models[2].Validation(ValidationSet, ValidationLabels, 'svm')
	pred04 = models[3].Validation(ValidationSet, ValidationLabels, 'svm')
	pred05 = models[4].Validation(ValidationSet, ValidationLabels, 'svm')
	pred06 = models[5].Validation(ValidationSet, ValidationLabels, 'svm')

	final_predictions = []

	for num in range(len(pred01)):
		votes = [0]*10
		votes[pred01[num]] += 1
		votes[pred02[num]] += 1
		votes[pred03[num]] += 1
		votes[pred04[num]] += 1
		votes[pred05[num]] += 1
		votes[pred06[num]] += 1

		label_voted = votes.index(max(votes))
		final_predictions.append(label_voted)

	return final_predictions

def __test_metrics(labels, predictions):
    print (metrics.accuracy_score(labels, predictions))
    print (metrics.classification_report(labels, predictions))
    print (metrics.confusion_matrix(labels, predictions).T)




