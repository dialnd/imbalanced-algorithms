import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def corrupt(x):
    """Take an input tensor and add uniform masking.

    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.

    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.
    """
    return tf.mul(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))

def _one_hot(label):
  """Convert label to One Hot encoding

  Parameters
  ----------

  label: Numpy 1-D array including numberic class labels

  Returns
  -------
  Numpy array of dimesions [n_samples,n_classes]
  """
  return pd.get_dummies(label)

def _read_split(file,read=0,oneHot=0):
  """

  Parameters
  ----------

  file: name of the csv file

  read: binary (To print dataset information)

  oneHot: binary (To return class labels as oneHot vectors)

  Returns
  -------

  train data, test data, train label, test label
  """
  df=pd.read_csv(file,skiprows=[1])
  df=df.rename(columns={df.columns[len(list(df))-1]:'Class'})
  df0=df[df['Class'] == 0]
  df1=df[df['Class'] == 1]
  imb=float(len(df0))/float(len(df1))
  if(read):
    print 'FEATURES : %d ROWS: %d ' %(( len(list(df))-1 ), len(df))
    print 'Imbalance Ratio: %f' %(imb)

  Xy0=df0.as_matrix().astype(np.float32)
  Xy1=df1.as_matrix().astype(np.float32)
  Xy=np.vstack((Xy0,Xy1))
  y=Xy[:,Xy.shape[1]-1]
  if(oneHot):
    y=_one_hot(y)
  X=np.delete(Xy,Xy.shape[1]-1,axis=1)
  return train_test_split(X,y, test_size=0.33, random_state=42)




def _reverse_one_hot(y):
  """Convert One Hot vectors to class labels
  Parameters
  ----------
  y: Numpy array with dimensions [n_samples,n_classes]

  Returns
  -------

  Class label
  """
  n_class=y.shape[1]
  label=np.arange(n_class)
  return np.dot(y,label)

def _class_split(trX,trY,oneHot=0):
  """Split the Class into Positive and Negative Class Samples
  Parameters
  ----------

  trX: Input Data, Numpy array (shape=[n_samples,n_features])
  trY: Class Label , Numpy Array
  OneHot: Binary (if class labels is in one hot encoding)
  """
  if(oneHot):
    Y=_reverse_one_hot(trY)
  else:
    Y=trY
  X=np.column_stack((trX,Y))
  class_0=X[np.where(X[:,X.shape[1]-1]==0)[0]]
  class_1=X[np.where(X[:,X.shape[1]-1]==1)[0]]
  class_0=np.delete(class_0,-1,axis=1)
  class_1=np.delete(class_1,-1,axis=1)
  return class_0,class_1

def _f_count(Y):
  '''Frequency Count of labels
  Y: Class Label. Numpy array
  '''
  if (len(Y.shape))==2:
    Y=_reverse_one_hot(Y)
  c = np.bincount(Y.astype(np.int32))
  ii = np.nonzero(c)[0]
  return zip(ii,c[ii])

def process_cm(confusion_mat, i=1, to_print=True):
    """Print TP,FP,TN,FN with confusion matrix as input 
    Parameters
    ----------

    confustion_mat: Numpy Array
    """
    TP = confusion_mat[i,i]  # correctly labeled as i
    FP = confusion_mat[:,i].sum() - TP  # incorrectly labeled as i
    FN = confusion_mat[i,:].sum() - TP  # incorrectly labeled as non-i
    TN = confusion_mat.sum().sum() - TP - FP - FN
    if to_print:
        print('TP: {}'.format(TP))
        print('FP: {}'.format(FP))
        print('FN: {}'.format(FN))
        print('TN: {}'.format(TN))
    return TP, FP, FN, TN

def factors(n):   
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))



def _read_dat(file,skip=13,read=1,oneHot=0):
  """Read .dat format from keel dataset
  Parameters
  ----------
  file : path to input file
  skip: rows to skip 
  read: binary (To print dataset information)

  oneHot: binary (To return class labels as oneHot vectors)

  Returns
  -------

  train data, test data, train label, test label
  """
  df=pd.read_csv(file,skiprows=skip,header=None)
  df=df.rename(columns={df.columns[len(list(df))-1]:'Class'})
  mapping = {'positive': 1, 'negative': 0}
  df=df.replace({'Class': mapping},regex=True)
  df0=df[df['Class'] == 0]
  df1=df[df['Class'] == 1]
  imb=float(len(df0))/float(len(df1))
  if(read):
    print 'FEATURES : %d ROWS: %d ' %(( len(list(df))-1 ), len(df))
    print 'Imbalance Ratio: %f' %(imb)

  Xy=df.as_matrix().astype(np.float32)
  y=Xy[:,Xy.shape[1]-1]
  if(oneHot):
    y=_one_hot(y)
  X=np.delete(Xy,Xy.shape[1]-1,axis=1)
  return train_test_split(X,y, test_size=0.33, random_state=42)

