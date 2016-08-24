import numpy as np
from sklearn.preprocessing import MinMaxScaler
from algorithms.utils import _read_split,_class_split,_one_hot
from algorithms.daego import DAEGO
from algorithms.daf import DAF
from algorithms.clf_utils import _clf_dtree,_clf_svm,_clf_mlp


#improvement of daego method using stacked encoders

#Flowchart
#1. Transform to higher dimension using DAF
#2. Generate Synthetic Samples using DAEGO
#3. Transform back to original dimension


trX, teX, trY, teY = _read_split("../datasets/nd-data/coil2000.csv",read=1,oneHot=0)

scaler=MinMaxScaler()
trX=scaler.fit_transform(trX)
print "Enter oversampling percent"
P=int(input())
X0,X1=_class_split(trX,trY)

print "X0 shape",X0.shape
print "X1 shape",X1.shape
print "Enter layer for DAF"
layer=input()
inp_shape=[trX.shape[1]]
layer_daf=inp_shape+layer
print "Enter batch Range for X0"
x0_batch=input()
print "Enter batch Range for X1"
x1_batch=input()
print "Enter Activation (1:sigmoid or 2:tanh)"
atcn=int(input())

if atcn==1:
	activation="sigmoid"
elif atcn==2:
	activation="tanh"
else:
	print "wrong activation"

print layer_daf,"DAF LAYER"

Z0_=DAF(X0,layer_daf,x0_batch,activation)
Z1_=DAF(X1,layer_daf,x1_batch,activation)

print "\n\n\nafter DAF encoding sample Z1_ shape:",Z1_.shape

print "Enter layer for DAEGO"
layer_daego=input()
inp_shape=[Z1_.shape[1]]
layer_daego=inp_shape+layer
print "Enter daego mini batch size"
daego_min_batch=input()
syn_Z=DAEGO(Z1_,layer_daego,P,daego_min_batch)

Z1_1=np.vstack((Z1_,syn_Z))

print Z1_1.shape,"after syn sample"

layer_daf.reverse()
X0_=DAF(Z0_,layer_daf,x0_batch,activation)
X1_=DAF(Z1_1,layer_daf,x1_batch,activation)


X1=np.column_stack((X1_,np.ones(X1_.shape[0])))
X0=np.column_stack((X0_,np.zeros(X0_.shape[0])))


Xy=np.vstack((X0,X1))
np.random.shuffle(Xy)
trY=Xy[:,Xy.shape[1]-1]
trX=np.delete(Xy,Xy.shape[1]-1,axis=1)


teX_scaled=scaler.fit_transform(teX)

print "\n\n\nWhether transform test data with daf (0/1)"
test_preprocess=input()

if (test_preprocess):
	layer_daf.reverse()
	layer_daf_test=layer_daf+layer_daf[::-1]
	print "Enter teX batch, teX Shape:",teX.shape
	teX_batch=input()
	teX=DAF(teX,layer_daf_test,teX_batch,activation)



_clf_dtree(trX,teX,trY,teY)
# _clf_svm(trX,teX,trY,teY)
_clf_mlp(trX,teX,trY,teY)


