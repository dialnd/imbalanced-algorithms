import numpy as np
from algorithms.sdae_s import sdae_syn
from algorithms.utils import _read_split,_class_split
from algorithms.clf_utils import _clf_dtree,_clf_svm,_clf_mlp,_clf_softmax
from sklearn.preprocessing import MinMaxScaler


#generate synthetic samples using deep sdae

# from algorithms.utils import _read_dat
# trX, teX, trY, teY = _read_dat(
# 	"dataset/page-blocks0.dat",skip=15,
# 	read=1,oneHot=0)

trX, teX, trY, teY = _read_split(
	"../datasets/nd-data/kddcup2004-protein-homology-train.csv",
	read=1,oneHot=0)

scaler=MinMaxScaler()

trX=scaler.fit_transform(trX)
teX=scaler.fit_transform(teX)

X0,X1=_class_split(trX,trY)
print "smaller class shape",X1.shape
print "Enter hidden layer for SDAE"
layer_sdae=input()
layer_sdae=layer_sdae+[X1.shape[1]]
print "Enter oversampling percent"
P=int(input())

syn_X=sdae_syn(X_s=X1,P=P,h_layer=layer_sdae,
	activations=['tanh' for i in range(len(layer_sdae))],
	noise='gaussian',epoch=[10000 for i in range(len(layer_sdae))],
	loss='rmse',batch_size=20)



X1=np.vstack((X1,syn_X))
X1=np.column_stack((X1,np.ones(X1.shape[0])))
X0=np.column_stack((X0,np.zeros(X0.shape[0])))
Xy=np.vstack((X0,X1))
np.random.shuffle(Xy)
trY=Xy[:,Xy.shape[1]-1]
trX=np.delete(Xy,Xy.shape[1]-1,axis=1)



_clf_dtree(trX,teX,trY,teY)
# _clf_svm(trX,teX,trY,teY)
_clf_mlp(trX,teX,trY,teY)