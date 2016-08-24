
import numpy as np
import tensorflow as tf
from utils import corrupt
import math
from sys import stderr
# %%
def autoencoder(dimensions):
    """Build a deep denoising autoencoder w/ tied weights.

    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    # input to the network
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')

    # Probability that we will corrupt input.
    # This is the essence of the denoising autoencoder, and is pretty
    # basic.  We'll feed forward a noisy input, allowing our network
    # to generalize better, possibly, to occlusions of what we're
    # really interested in.  But to measure accuracy, we'll still
    # enforce a training signal which measures the original image's
    # reconstruction cost.
    #
    # We'll change this to 1 during training
    # but when we're ready for testing/production ready environments,
    # we'll put it back to 0.
    corrupt_prob = tf.placeholder(tf.float32, [1])
    current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)

    # Build the encoder
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    # latent representation
    z = current_input
    encoder.reverse()
    # Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.sqrt(tf.reduce_mean(tf.square(y - x)))
    return {'x': x, 'z': z, 'y': y,
            'corrupt_prob': corrupt_prob,
            'cost': cost}



def _encoder_transform(X_s,layers,batch_range):
    """
    Parameters:
    ----------

    X_s: input data
    layers: neuron layers (input shape + hidden layers)
    batch_range: size of minibatch

    Returs:

    Input data with fetures as most latent representation

    """
    ae= autoencoder(dimensions=layers)
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    n_epoch=100
    for epoch_i in range(n_epoch):
        for start, end in zip(range(0, len(X_s), batch_range),range(batch_range, len(X_s), batch_range)):
            input_ = X_s[start:end]
            sess.run(optimizer, feed_dict={ae['x']: input_, ae['corrupt_prob']: [1.0]})
        s="\r Epoch: %d Cost: %f"%(epoch_i, sess.run(ae['cost'], 
            feed_dict={ae['x']: X_s, ae['corrupt_prob']: [1.0]}))  
        stderr.write(s)
        stderr.flush()
    Z_0 = sess.run(ae['z'], feed_dict={ae['x']: X_s, ae['corrupt_prob']: [0.0]})
    sess.close()
    return Z_0





