import numpy as np
import warnings
from algorithms.utils import _read_split,_class_split
from algorithms.clf_utils import _clf_dtree,_clf_svm,_clf_mlp
from algorithms.smote import SMOTE
from deepautoencoder import StackedAutoEncoder
from sklearn.preprocessing import MinMaxScaler as scaler



# trX, teX, trY, teY = _read_split(
# 	"../datasets/nd-data/coil2000.csv",
# 	read=1,oneHot=0)

#Integrating smote with daego

#perform smote at the intermediate stage of training via stacked denoising encoder

from algorithms.utils import _read_dat
trX, teX, trY, teY = _read_dat(
	"dataset/page-blocks0.dat",skip=15,
	read=1,oneHot=0)


scaler=scaler()
trX=scaler.fit_transform(trX)
teX=scaler.fit_transform(teX)
from mlxtend.tf_classifier import TfSoftmaxRegression
trY=trY.astype(int)

print trX.shape[1],"Input Feature Space"

print "Enter Layers"
layer=input()
print "Enter the leyer no after smote to be performed"
l_s=int(input())
l_encoder=layer[:l_s]

model_bs = StackedAutoEncoder(dims=l_encoder, activations=['tanh' for i in range(len(l_encoder))], noise='gaussian', 
	epoch=[10000 for i in range(len(l_encoder))],loss='rmse', 
	lr=0.007, batch_size=20, print_step=2000)

S1=model_bs.fit_transform(trX)

X0,X1=_class_split(S1,trY)
print X0.shape[1],"Feature Space Before smote"
warnings.filterwarnings("ignore", category=DeprecationWarning)
print "Enter oversampling percent"
P=int(input())
syn_X=SMOTE(X1, P, 5)
X1=np.vstack((X1,syn_X))
X1=np.column_stack((X1,np.ones(X1.shape[0])))
X0=np.column_stack((X0,np.zeros(X0.shape[0])))
Xy=np.vstack((X0,X1))
np.random.shuffle(Xy)
trY=Xy[:,Xy.shape[1]-1]
S1_smote=np.delete(Xy,Xy.shape[1]-1,axis=1)
trY=trY.astype(int)

l_decoder=layer[l_s:]

model_as = StackedAutoEncoder(dims=l_decoder, activations=['tanh' for i in range(len(l_decoder))], noise='gaussian', 
	epoch=[10000 for i in range(len(l_decoder))],loss='rmse', lr=0.007, 
	batch_size=20, print_step=2000)

S=model_as.fit_transform(S1_smote)

model_test=StackedAutoEncoder(dims=layer, activations=['tanh' for i in range(len(l_encoder))]+['tanh' for i in range(len(l_decoder))], noise='gaussian', 
	epoch=[10000 for i in range(len(layer))],loss='rmse', lr=0.007, 
	batch_size=20, print_step=2000)
teX=model_test.fit_transform(teX)



_clf_dtree(S,teX,trY,teY)
_clf_svm(S,teX,trY,teY)
_clf_mlp(S,teX,trY,teY)