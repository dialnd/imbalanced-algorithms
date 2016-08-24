import numpy as np
from sklearn.metrics import precision_score,roc_auc_score,recall_score,confusion_matrix
from algorithms.utils import _read_split,_class_split,_one_hot
from algorithms.daego import DAEGO
from algorithms.clf_utils import _clf_dtree,_clf_svm,_clf_mlp

trX, teX, trY, teY = _read_split(
	"datasets/boundary.csv",
	read=1,oneHot=0)

#Test the classifier performance using synthetic samples generated via daego 
#link to paper: http://www.site.uottawa.ca/~nat/Papers/DEAGO-PID3925613.pdf
# from algorithms.utils import _read_dat
# trX, teX, trY, teY = _read_dat(
# 	"dataset/page-blocks0.dat",skip=15,
# 	read=1,oneHot=0)

	
X0,X1=_class_split(trX,trY,oneHot=0)


print "smaller class shape",X1.shape
print "Enter layer"
layer_daego=input()
print "Enter oversampling percent"
P=int(input())
print "Enter batch Range"
batch_range=input()

inp_shape=[X1.shape[1]]
layer_daego=inp_shape+layer_daego
syn_X=DAEGO(X1,layer_daego,P,batch_range)

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