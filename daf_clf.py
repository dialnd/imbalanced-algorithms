import numpy as np
from algorithms.utils import _read_split,_class_split
from algorithms.sdae import sdae
from algorithms.clf_utils import _clf_dtree,_clf_svm,_clf_mlp
from sklearn.preprocessing import MinMaxScaler

trX, teX, trY, teY = _read_split("../datasets/nd-data/boundary.csv",read=1)

#implementation of dual encoded features algorithm
#link to paper: http://www.sciencedirect.com/science/article/pii/S0031320316301303


# from algorithms.utils import _read_dat
# trX, teX, trY, teY = _read_dat(
# 	"dataset/page-blocks0.dat",skip=15,
# 	read=1,oneHot=0)


def _sdae_module(X,Y,layer,batch_range):
	scaler=MinMaxScaler()
	X=scaler.fit_transform(X)
	X1=sdae(X,layer,batch_range,"sigmoid")
	X2=sdae(X,layer,batch_range,"tanh")
	Xy1=np.column_stack((X1,Y))
	Xy2=np.column_stack((X2,Y))
	Xy=np.vstack((Xy1,Xy2))
	np.random.shuffle(Xy)
	y=Xy[:,Xy.shape[1]-1]
	X=np.delete(Xy,Xy.shape[1]-1,axis=1)
	return X,y



#preprocessing


print "trX shape",trX.shape
print "teX shape",teX.shape
print "Enter layer for sdae"
layer=input()
inp_shape=[trX.shape[1]]
layer=inp_shape+layer
print "Enter batch Range for training"
train_batch=input()
print "Enter batch Range for testing"
test_batch=input()

trX,trY=_sdae_module(trX,trY,layer,train_batch)
teX,teY=_sdae_module(teX,teY,layer,test_batch)

_clf_dtree(trX,teX,trY,teY)
# _clf_svm(trX,teX,trY,teY)
_clf_mlp(trX,teX,trY,teY)

