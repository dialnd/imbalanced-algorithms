import numpy as np
import warnings
from algorithms.utils import _read_split,_class_split
from algorithms.clf_utils import _clf_dtree,_clf_svm,_clf_mlp
from algorithms.smote import SMOTE


#Test the classifier performance using synthetic samples generated via SMOTE 
#link to paper: https://www.jair.org/media/953/live-953-2037-jair.pdf

from algorithms.utils import _read_dat
trX, teX, trY, teY = _read_dat(
	"datasets/page-blocks0.dat",skip=15,
	read=1,oneHot=0)

# trX, teX, trY, teY = _read_split(
# 	"../datasets/nd-data/boundary.csv",
# 	read=1,oneHot=0)


X0,X1=_class_split(trX,trY,oneHot=0)

warnings.filterwarnings("ignore", 
	category=DeprecationWarning)
print "Enter oversampling percent"
P=int(input())
syn_X=SMOTE(X1, P, 5)
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


