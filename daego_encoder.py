import numpy as np
from sklearn.preprocessing import MinMaxScaler
from algorithms.utils import _read_split,_class_split,_one_hot
from algorithms.daego import DAEGO
from algorithms.clf_utils import _clf_dtree,_clf_svm,_clf_mlp
from algorithms.encoder import _encoder_transform

# trX, teX, trY, teY = _read_split("../datasets/nd-data/boundary.csv",read=1,oneHot=0)

from algorithms.utils import _read_dat
trX, teX, trY, teY = _read_dat(
	"dataset/page-blocks0.dat",skip=15,
	read=1,oneHot=0)

#improvement of daego method using denoising encoders

#Flowchart
#1. Transform to higher dimension using encoders
#2. Generate Synthetic Samples using DAEGO
#3. Transform back to original dimension

scaler=MinMaxScaler()
trX=scaler.fit_transform(trX)
print "Enter oversampling percent"
P=int(input())
X0,X1=_class_split(trX,trY)

print "X0 shape",X0.shape
print "X1 shape",X1.shape
print "Enter layer for Encoder"
layer=input()
inp_shape=[trX.shape[1]]
layer_enc=inp_shape+layer
print "Enter batch Range for X0"
x0_batch=input()
print "Enter batch Range for X1"
x1_batch=input()


print layer_enc,"Encoder LAYER"

Z0_=_encoder_transform(X0,layer_enc,x0_batch)
Z1_=_encoder_transform(X1,layer_enc,x1_batch)

print "\n\n\nafter sdae encoding sample Z1_ shape:",Z1_.shape

print "Enter layer for DAEGO"
layer_daego=input()
inp_shape=[Z1_.shape[1]]
layer_daego=inp_shape+layer
print "Enter daego mini batch size"
daego_min_batch=input()

syn_Z=DAEGO(Z1_,layer_daego,P,daego_min_batch)

Z1_1=np.vstack((Z1_,syn_Z))

print Z1_1.shape,"after syn sample"

layer_enc.reverse()
X0_=_encoder_transform(Z0_,layer_enc,x0_batch)
X1_=_encoder_transform(Z1_1,layer_enc,x1_batch)


X1=np.column_stack((X1_,np.ones(X1_.shape[0])))
X0=np.column_stack((X0_,np.zeros(X0_.shape[0])))


Xy=np.vstack((X0,X1))
np.random.shuffle(Xy)
trY=Xy[:,Xy.shape[1]-1]
trX=np.delete(Xy,Xy.shape[1]-1,axis=1)


teX=scaler.fit_transform(teX)
print "\n\n\nWhether preprocess test data with encoder (0/1)"
test_preprocess=input()

if (test_preprocess):
	layer_enc.reverse()
	layer_enc_test=layer_enc+layer_enc[::-1]
	print "Enter teX batch, teX Shape:",teX.shape
	teX_batch=input()
	teX=_encoder_transform(teX,layer_enc_test,teX_batch)


_clf_dtree(trX,teX,trY,teY)
# _clf_svm(trX,teX,trY,teY)
_clf_mlp(trX,teX,trY,teY)