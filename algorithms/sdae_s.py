import numpy as np
import numpy.linalg as LA
from deepautoencoder import StackedAutoEncoder
from sklearn.preprocessing import MinMaxScaler

def sdae_syn(X_s,P,h_layer,activations,noise,epoch,loss,batch_size):
	"""Generate synthetic samples using stacked De-noising Encoders
	Parameters
	----------
	X_s: positive class sample (Numpy Array) (Input Must be in within range of 0 to 1)
	P: Over Sampling Percentage
	h_layer: hidden layer (list)
	activation: activation functions list (same length as hidden layer)
	noise : [None,Gaussian,mask]
	epoch: epoch for each layer (list with same size as hidden layer)
	loss: 'rmse' or 'cross-entropy'
	batch_size = mini_batch size

	For more detaisl on input parameters https://github.com/rajarsheem/libsdae 
	"""
	n_samples=int(X_s.shape[0]*P/100)
	print "generating %d samples" %(n_samples)
	X_init=np.random.standard_normal(size=(n_samples,X_s.shape[1]))
	scaler=MinMaxScaler()
	X_init=scaler.fit_transform(X_init)
	model = StackedAutoEncoder(dims=h_layer, activations=activations, noise=noise, 
		epoch=epoch,loss=loss, 
		batch_size=batch_size, lr=0.007, print_step=2000)
	model.fit(X_s)
	syn_Z=model.transform(X_init)
	return syn_Z


