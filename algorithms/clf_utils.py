import numpy as np
from sklearn import tree
from sklearn.metrics import precision_score,roc_auc_score,recall_score,confusion_matrix
from utils import _read_split,_class_split,_one_hot,_f_count,process_cm,factors
from mlxtend.tf_classifier import TfMultiLayerPerceptron,TfSoftmaxRegression
from sklearn.svm import SVC


def _clf_dtree(trX,teX,trY,teY):
	print "DECISION TREE"
	print _f_count(teY),"test f count"
	clf = tree.DecisionTreeRegressor()
	clf = clf.fit(trX, trY)
	pred=clf.predict(teX)
	pred=pred.astype(np.int32)
	teY=teY.astype(np.int32)
	print _f_count(pred),"pred f count"
	conf_mat=confusion_matrix(teY, pred)

	process_cm(conf_mat, to_print=True)
	print precision_score(teY,pred),"Precision Score"
	print recall_score(teY,pred),"Recall Score"
	print roc_auc_score(teY,pred), "ROC_AUC"

def _clf_svm(trX,teX,trY,teY):
	print "SVM"
	print _f_count(teY),"test f count"
	clf = SVC(random_state=0, probability=True)
	clf = clf.fit(trX, trY)
	pred=clf.predict(teX)
	pred=pred.astype(np.int32)
	teY=teY.astype(np.int32)
	print _f_count(pred),"pred f count"
	conf_mat=confusion_matrix(teY, pred)

	process_cm(conf_mat, to_print=True)
	print precision_score(teY,pred),"Precision Score"
	print recall_score(teY,pred),"Recall Score"
	print roc_auc_score(teY,pred), "ROC_AUC"

def _clf_mlp(trX,teX,trY,teY):
	print "MLP"
	print trX.shape,"trX shape"
	print "Enter Layer for MLP"
	layer=input()
	# print "enter delIdx"
	# delIdx=input()
	# while(delIdx):
	# 	trX=np.delete(trX,-1,axis=0)
	# 	trY=np.delete(trY,-1,axis=0)
	# 	delIdx=delIdx-1
	print "factors",factors(trX.shape[0])	
	teY=teY.astype(np.int32)
	trY=trY.astype(np.int32)
	print trX.shape,"trX shape"
	print "enter no of mini batch"
	mini_batch=int(input())
	mlp = TfMultiLayerPerceptron(eta=0.01, 
                             epochs=100, 
                             hidden_layers=layer,
                             activations=['relu' for i in range(len(layer))],
                             print_progress=3, 
                             minibatches=mini_batch, 
                             optimizer='adam',
                             random_seed=1)
	mlp.fit(trX,trY)
	pred=mlp.predict(teX)
	print _f_count(teY),"test f count"
	pred=pred.astype(np.int32)
	print _f_count(pred),"pred f count"
	conf_mat=confusion_matrix(teY, pred)
	process_cm(conf_mat, to_print=True)
	print precision_score(teY,pred),"Precision Score"
	print recall_score(teY,pred),"Recall Score"
	print roc_auc_score(teY,pred), "ROC_AUC"


def _clf_softmax(trX,teX,trY,teY):
	print "factors",factors(trX.shape[0])
	print "enter no of mini batch"
	trY=trY.astype(int)
	teY=teY.astype(int)
	mini_batch=int(input())
	clf = TfSoftmaxRegression(eta=0.75, 
                         epochs=100, 
                         print_progress=True, 
                         minibatches=mini_batch, 
                         random_seed=1)
	clf.fit(trX, trY)
	pred=clf.predict(teX)
	print _f_count(teY),"test f count"
	pred=pred.astype(np.int32)
	print _f_count(pred),"pred f count"
	conf_mat=confusion_matrix(teY, pred)
	process_cm(conf_mat, to_print=True)
	print precision_score(teY,pred),"Precision Score"
	print recall_score(teY,pred),"Recall Score"
	print roc_auc_score(teY,pred), "ROC_AUC"





