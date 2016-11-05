import argparse
import copy
import math
import random
import sys

import numpy as np
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

DEBUG = False

def linear(input_, output_size, scope=None, stddev=0.5, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

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

def corrupt_gaussian(x, std=1.0):
    """Take an input tensor and add Gaussian noise.

    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.
    std: Integer
        The desired standard deviation of noise.

    Returns
    -------
    x_corrupted : Tensor
        Adds Gaussian noise to the input with mean zero and stddev std.
    """
    return tf.add(x, tf.random_normal(shape=tf.shape(x),
                                      mean=0.0,
                                      stddev=std))

def autoencoder(layer_dim, std=1.0):
    """Build a deep denoising autoencoder w/ tied weights.

    Parameters
    ----------
    layer_dim : list, optional
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

    Notes
    -----
    Implementation based on:
        https://github.com/pkmital/tensorflow_tutorials/blob/master/python/08_denoising_autoencoder.py
    """
    # Input to the network.
    x = tf.placeholder(tf.float32, [None, layer_dim[0]], name='x')

    # Probability that we will corrupt input.
    # This is the essence of the denoising autoencoder, and is pretty
    # basic. We'll feed forward a noisy input, allowing our network
    # to generalize better, possibly, to occlusions of what we're
    # really interested in. But to measure accuracy, we'll still
    # enforce a training signal which measures the original image's
    # reconstruction cost.
    #
    # We'll change this to 1 during training
    # but when we're ready for testing/production ready environments,
    # we'll put it back to 0.
    corrupt_prob = tf.placeholder(tf.float32, [1])
    #current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)
    current_input = corrupt_gaussian(x, std=std) * corrupt_prob + x * (1 - corrupt_prob)

    # Build the encoder.
    encoder = []
    for layer_i, n_output in enumerate(layer_dim[1:]):
        n_input = int(current_input.get_shape()[1])
        #W = tf.Variable(
        #    tf.random_uniform([n_input, n_output],
        #                      -1.0 / math.sqrt(n_input),
        #                      1.0 / math.sqrt(n_input)))
        #b = tf.Variable(tf.zeros([n_output]))
        layer_bn = tf.contrib.layers.batch_norm(current_input, 
            scope="A_{0}".format(layer_i))
        lin, W, b = linear(layer_bn, n_output, 
            scope="A_{0}".format(layer_i), with_w=True)
        encoder.append(W)
        #output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        output = tf.nn.tanh(lin)

        current_input = output

    # Latent representation.
    z = current_input
    encoder.reverse()

    # Build the decoder using the same weights.
    for layer_i, n_output in enumerate(layer_dim[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

    # Now have the reconstruction through the network.
    y = current_input

    # Cost function measures pixel-wise difference.
    #cost = tf.sqrt(tf.reduce_mean(tf.square(y - x)))
    cost = tf.reduce_mean(tf.square(y - x))

    return {'x': x, 'z': z, 'y': y,
            'corrupt_prob': corrupt_prob,
            'cost': cost}

def minibatch(layer_input, num_kernels=5, kernel_dim=3):
    x = linear(layer_input, num_kernels * kernel_dim, 
        scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - 
    tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    eps = tf.expand_dims(np.eye(int(layer_input.get_shape()[0]), 
        dtype=np.float32), 1)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2) + eps
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat(1, [layer_input, minibatch_features])

class Dataset(object):
    def __init__(self, data):
        self.scaler = MinMaxScaler((-1, 1))
        self.data = self.scaler.fit_transform(data)

        self.data_norm = normalize(self.data, axis=0)
        self.norm_param = [np.linalg.norm(x) for x in self.data.T]

        self.scaler2 = StandardScaler()
        self.data = self.scaler2.fit_transform(self.data.astype(float))

class DAESO(object):
    """Denoising Autoencoder (DAE) approach to synthetic oversampling for 
    modeling the minority class and synthesizing new instances to balance 
    the training set.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Matrix containing the data to be learned.
    num_epochs : int
        Number of epochs to train.
    batch_size : int
        The number of instances per batch.
    minibatch : boolean
        Use minibatches.
    dimensions : int
        The number of units per hidden layer.
    layers : int
        The number of hidden layers.
    stddev : float
        The type of SMOTE algorithm to use one of the following options:
        'regular', 'borderline1', 'borderline2', 'svm'.

    Notes
    -----
    See the original papers for more details:
        [1] C. Bellinger, N. Japkowicz, and C. Drummond. "Synthetic 
            Oversampling for Advanced Radioactive Threat Detection". IEEE 14th 
            International Conference on Machine Learning and Applications 
            (ICMLA), 2015.
        [2] C. Bellinger, C. Drummond, and N. Japkowicz. "Beyond the 
            Boundaries of SMOTE." Joint European Conference on Machine Learning 
            and Knowledge Discovery in Databases". Springer International 
            Publishing, 2016.
    """
    def __init__(self, data, num_epochs, batch_size, minibatch, 
                 layers, dimensions, stddev):
        dataset = Dataset(data)
        self.data = dataset.data

        self.scaler = dataset.scaler
        self.scaler2 = dataset.scaler2
        self.data_norm = dataset.data_norm
        self.norm_param = dataset.norm_param

        self.learning_rate = 0.001
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.minibatch = False
        self.dae_hidden_dim = [dimensions] * layers
        self.std = stddev

        self._create_model()

    def _create_model(self):
        self.data_norm = normalize(self.data, axis=0)
        self.norm_param = [np.linalg.norm(x) for x in self.data.T]
        self.scaler2 = StandardScaler()
        self.data = self.scaler2.fit_transform(self.data.astype(float))

        # Create a denoising autoencoder network dae.
        layer_dim = np.append(np.array(self.data.shape[1]), self.dae_hidden_dim)
        self.dae = autoencoder(layer_dim=layer_dim, sig=self.std)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.dae['cost'])

    def train(self, session):
        tf.initialize_all_variables().run()

        if DEBUG:
            scaler = MinMaxScaler((-.5,.5))
            data_tr = scaler.fit_transform(self.data)
            z0_data = np.random.uniform(data_tr.min(axis=0)[0], 
                data_tr.max(axis=0)[0], [data_tr.shape[0], 1]).astype(np.float32)
            z1_data = np.random.uniform(data_tr.min(axis=0)[1], 
                data_tr.max(axis=0)[1], [data_tr.shape[0], 1]).astype(np.float32)
            z_data = np.concatenate((z0_data, z1_data), axis=1)
            
            import matplotlib
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1,1)
            ax.scatter(z_data[:,0], z_data[:,1], color=[1,0,0,1], s=3)
            dae = self.A.eval({self.z: z_data})
            ax.scatter(dae[:,0], dae[:,1], color=[0,0,1,1], s=3)
            fig.canvas.draw()

        # Train the network on the normalized training data.
        for epoch in range(self.num_epochs):
            batch_idxs = self.data.shape[0] // self.batch_size
            for idx in range(0, batch_idxs):
                batch = np.array(
                    self.data[idx*self.batch_size:(idx+1)*self.batch_size]).astype(np.float32)
                session.run(self.opt, feed_dict={
                    self.dae['x']: batch, self.dae['corrupt_prob']: [.1]
                    })

        return session

    def gen_samples(self, session, N):
        noise = np.random.normal(loc=0.0, scale=self.std, size=(N, self.data.shape[1]))
        # Apply noise the sample initiation set X_init.
        Z = self.data + noise

        # Map X_norm to the induced manifold via dae.
        X_recon = session.run(self.dae['y'], 
            feed_dict={self.dae['x']: Z, self.dae['corrupt_prob']: [0.0]})
        # Denormalize the mapped synthetic instances.
        X_recon = np.multiply(X_recon, self.norm_param)
        X_recon = self.scaler2.inverse_transform(X_recon)
        data_syn = self.scaler.inverse_transform(X_recon)

        if DEBUG:
            data_itr = self.scaler.inverse_transform(self.data)
            g_itr = self.scaler.inverse_transform(g)
            
            import matplotlib
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1,1)
            ax.scatter(data_itr[:,0], data_itr[:,1], color=[1,0,0,1], s=3)
            ax.scatter(g_itr[:,0], g_itr[:,1], color=[0,0,1,1], s=3)
            fig.canvas.draw()

        return self.scaler.inverse_transform(g)

def main(data, N, args):
    tf.reset_default_graph()
    session = tf.InteractiveSession()

    model = GAN(dataset,
        args.num_epochs,
        args.batch_size,
        args.minibatch,
        args.layers,
        args.dims,
        args.stddev
        )
    session, _ = model.train(session)
    samples = model.gen_samples(session, N)

    session.close()

    return samples

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int, default=1000,
                        help='The number of training epochs to take.')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='The batch size.')
    parser.add_argument('--minibatch', type=bool, default=False,
                        help='Use minibatch discrimination.')
    parser.add_argument('--layers', type=int, default=2,
                        help='The number of hidden layers.')
    parser.add_argument('--dims', type=int, default=2,
                        help='The number of units per hidden layer.')
    parser.add_argument('--stddev', type=int, default=2,
                        help='The standard deviation for the initialization noise.')
    parser.add_argument('--log-every', type=int, default=10,
                        help='Print loss after this many steps.')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
