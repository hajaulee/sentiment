import numpy as np
from matplotlib import pyplot as plt 

def show_training_history(file_path):
	histories=np.loadtxt(file_path, delimiter=",")
	plt.plot(histories[:, 1])
	plt.plot(histories[:, 3])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	
	# summarize history for loss
	plt.plot(histories[:, 0])
	plt.plot(histories[:, 2])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()