import numpy as np
from encoder import autoencoder
import numpy.linalg as LA
from sklearn.preprocessing import StandardScaler as StdScaler
from sklearn.preprocessing import normalize as norm
import tensorflow as tf
from sys import stderr

def DAEGO(X_s,H,P,batch_range):
	"""
	Parameters
	----------

	X_s: small class features

	H : layers (first layers shoud have same neurons as number of features)

	P : percent oversampling

	batch_range : size of minibatch


	Returns
	-------

	syn_Z: synthetic sample with same number of features as smaller class
	"""

	#normalization
	scaler=StdScaler()
	x_tr=scaler.fit_transform(X_s.astype(float))
	x_norm=norm(x_tr,axis=0)

	n_samples=int(X_s.shape[0]*P/100)
	print "generating %d samples" %(n_samples)

	norm_param=[LA.norm(x) for x in x_tr.T]
	X_init=np.random.standard_normal(size=(n_samples,X_s.shape[1]))
	x_init_tr=scaler.transform(X_init)
	x_ini_norm=norm(x_init_tr)
	ae=autoencoder(dimensions=H)
	learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	n_epoch=100
	for epoch_i in range(n_epoch):
	    for start, end in zip(range(0, len(x_norm), batch_range),range(batch_range, len(x_norm), batch_range)):
	        input_ = x_norm[start:end]
	        sess.run(optimizer, feed_dict={ae['x']: input_, ae['corrupt_prob']: [1.0]})
	    s="\r Epoch: %d Cost: %f"%(epoch_i, sess.run(ae['cost'], 
	    	feed_dict={ae['x']: X_s, ae['corrupt_prob']: [1.0]}))
	    stderr.write(s)
	    stderr.flush()
	x_init_encoded = sess.run(ae['y'], feed_dict={ae['x']: x_ini_norm, ae['corrupt_prob']: [0.0]})
	sess.close()
	x_init_norminv=np.multiply(x_init_encoded,norm_param)
	syn_Z=scaler.inverse_transform(x_init_norminv)
	return syn_Z


