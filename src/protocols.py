

def protocol_0(images, labels, query_image, labels_query, models):

	pred01 = models[0].kNN(images, labels, query_image, 10)
	pred02 = models[1].kNN(images, labels, query_image, 10)
	pred03 = models[2].kNN(images, labels, query_image, 10)
	pred04 = models[3].kNN(images, labels, query_image, 10)
	pred05 = models[4].kNN(images, labels, query_image, 10)
	pred06 = models[5].kNN(images, labels, query_image, 10)


def protocol_1(images, labels, query_image, labels_query, models):

	pred01 = models[0].kNN(images, labels, query_image, 10, kNN_type='cossine')
	pred02 = models[1].kNN(images, labels, query_image, 10, kNN_type='cossine')
	pred03 = models[2].kNN(images, labels, query_image, 10, kNN_type='cossine')
	pred04 = models[3].kNN(images, labels, query_image, 10, kNN_type='cossine')
	pred05 = models[4].kNN(images, labels, query_image, 10, kNN_type='cossine')
	pred06 = models[5].kNN(images, labels, query_image, 10, kNN_type='cossine')
