import pickle
import numpy as np
from keras.datasets import cifar10
from matplotlib import pylab as plt
import cv2

def unpickle(filename):
	with open(filename, 'rb') as fp:
		dict = pickle.load(fp, encoding='bytes')
	
	data = dict[b'data']
	labels = dict[b'labels']
	
	data = np.array(data).reshape((len(labels), 3, 32, 32))
	labels = np.array(labels).reshape(len(labels))
	return data, labels


	(x_train, y_train), (x_eval, y_eval) = cifar10.load_data()
	return (x_train, y_train), (x_eval, y_eval)
