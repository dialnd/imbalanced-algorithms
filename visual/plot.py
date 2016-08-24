import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from utils import _class_split
def plot_data(X,y):
	"""split data into pos and neg classes transform into 2D using PCA and plot
	"""
	X0,X1=_class_split(X,y)
	pca=PCA(n_components=2)
	x0=pca.fit_transform(X0)
	x1=pca.fit_transform(X1)
	plt.title("PCA of two classes")
	plt.scatter(x0[:,0],x0[:,1],marker='o', c="#00ffff",label="neg Class")
	plt.scatter(x1[:,0],x1[:,1],marker='o', c="r",label="pos Class")
	plt.legend()
	plt.show()

def plot_syn(X1,S1,labl):
	"""To analyze synthetic samples generated
	"""
	pca=PCA(n_components=2)
	x0=pca.fit_transform(X1)
	x1=pca.fit_transform(S1)
	plt.title("synthetic data distribution")
	plt.scatter(x0[:,0],x0[:,1],marker='o', c="#00ffff",label="raw Data")
	plt.scatter(x1[:,0],x1[:,1],marker='o', c="r",label=labl)
	plt.legend()
	plt.show()

def plot_X(X):
	"""Plot individual class or any multidimensional data in 2D
	"""
	pca=PCA(n_components=2)
	x0=pca.fit_transform(X)
	plt.title("PLOT X")
	plt.scatter(x0[:,0],x0[:,1],marker='o', c="#00ffff",label="raw Data")
	plt.legend()
	plt.show()