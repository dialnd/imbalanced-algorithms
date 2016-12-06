import argparse

import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)

def init_xavier(fan, constant=1): 
    """Xavier initialization of network weights."""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    fan_in, fan_out = fan[0], fan[1]
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

def binary_crossentropy(output, target):
    """Compute the binary cross-entropy.

    Add 1e-10 to avoid evaluation of log(0.0).
    """
    #return -tf.reduce_sum(target * tf.log(output + 1e-10) 
    #       + (1-target) * tf.log(1-output + 1e-10), 1)
    return -tf.reduce_sum(target * tf.log(tf.clip_by_value(output, 1e-10, 1.0)) 
        + (1-target) * tf.log(tf.clip_by_value(1-output, 1e-10, 1.0)))

def lrelu(X, leak=0.2, name='lrelu'):
    """Leaky rectified linear unit (LReLU)."""
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * X + f2 * abs(X)

def linear(input_, output_size, scope=None, stddev=0.5, bias_start=0.0, with_w=False):
    """Compute the linear dot product with the input and its weights plus bias.

    Parameters
    ----------
    input_ : Tensor
        Tensor on which to apply dot product.
    output_size : int
        Number of outputs.

    Returns
    -------
    Tensor
        Linear dot product. 
    """
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def binomial(shape=[1], p=0.5, dtype='float32'):
    """Generate a binomial distribution.

    Parameters
    ----------
    shape : list
        Shape of binomial distribution.
    p : float
        Probability level for dropping an element.

    Returns
    -------
    Tensor
        Binomial distribution.
    """
    dist = tf.random_uniform(shape=shape, minval=0, maxval=1, dtype='float32')
    return tf.select(tf.less(dist, tf.fill(shape, p)), tf.ones(shape, dtype=dtype), 
           tf.zeros(shape, dtype=dtype))

def binomial_vec(p_vec, shape=[1], dtype='float32'):
    """Generate a binomial distribution based on a vector of probabilities.

    Parameters
    ----------
    p_vec : array
        Probability vector.
    shape : list
        Shape of binomial distribution.

    Returns
    -------
    Tensor
        Binomial distribution.
    """
    dist = tf.random_uniform(shape=shape, minval=0, maxval=1, dtype='float32')
    return tf.select(tf.less(dist, p_vec), tf.ones(shape, dtype=dtype), 
           tf.zeros(shape, dtype=dtype))

def dropout(X, p=0.5):
    """Dropout function.

    Parameters
    ----------
    X : Tensor
        Tensor on which to apply dropout.
    p : float
        Probability level for dropping an element (used in binomial distribution).

    Returns
    -------
    Tensor
        Tensor with dropout applied.
    """
    noise = binomial(shape=tf.shape(X), p=p)
    return tf.div(tf.mul(X, noise), p)

def corrupt_salt_and_pepper(X, rate=0.3):
    """Take an input tensor and add salt-and-pepper noise.

    Parameters
    ----------
    X : Tensor/Placeholder
        Input to corrupt.
    rate : float
        The desired rate of corruption.

    Returns
    -------
    x_corrupted : Tensor
        Rate percentage of values corrupted.
    """
    a = binomial(shape=tf.shape(X), p=1-rate)
    b = binomial(shape=tf.shape(X), p=0.5)
    z = tf.zeros(tf.shape(X), dtype='float32')
    c = tf.select(tf.equal(a, z), b, z)
    return tf.add(tf.mul(X, a), c)

def corrupt_gaussian(X, std=1.0):
    """Take an input tensor and add Gaussian noise.

    Parameters
    ----------
    X : Tensor/Placeholder
        Input to corrupt.
    std: float
        The desired standard deviation of noise.

    Returns
    -------
    x_corrupted : Tensor
        Adds Gaussian noise to the input with mean zero and standard deviation std.
    """
    return tf.add(X, tf.random_normal(shape=tf.shape(X),
                                      mean=0.0,
                                      stddev=std))

class DAE(object):
    """Denoising Autoencoder (DAE) implemented using TensorFlow.

    Parameters
    ----------
    num_epochs : int
        Passes over the training dataset.
    batch_size : int
        Size of minibatches for stochastic optimizers.
    hidden_dim : list
        Number of units per hidden layer for encoder/decoder.
    n_input : int
        Number of inputs to initial layer.
    corrupt_prob : float
        Probability of generating corrupted values.
    corrupt_std : float
        Standard deviation of corrupted values (gaussian).
    corrupt_type : str
        Type of corrupting function (gaussian or salt_and_pepper).
    walkbacks : int
        Number of walkbacks to use.
    transfer_fct : object
        Transfer function for hidden layers.
    W_init_fct : object
        Initialization function for weights.
    b_init_fct : object
        Initialization function for biases.
    learning_rate : float
        Learning rate schedule for weight updates.
    log_every : int
        Print loss after this many steps.

    Notes
    -----
    See the original paper for more details:
        [1] P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, and P.-A. Manzagol. 
            "Stacked Denoising Autoencoders: Learning Useful Representations in 
            a Deep Network with a Local Denoising Criterion". Journal of 
            Machine Learning Research (JMLR), 2010.

    Implements pseudo-Gibbs sampling and walkback training based on:
        [2] Y. Bengio, L. Yao, G. Alain, and P. Vincent. "Generalized Denoising 
            Auto-Encoders as Generative Models". Advances in Neural Information 
            Processing Systems 26 (NIPS), 2013.

    Generalized to oversampling problems based on the following work:
        [3] C. Bellinger, C. Drummond, and N. Japkowicz. "Beyond the 
            Boundaries of SMOTE". Joint European Conference on Machine Learning 
            and Knowledge Discovery in Databases (ECML-PKDD), 2016.
        [4] C. Bellinger, N. Japkowicz, and C. Drummond. "Synthetic 
            Oversampling for Advanced Radioactive Threat Detection". IEEE 14th 
            International Conference on Machine Learning and Applications 
            (ICMLA), 2015.

    Based on related code:
        - https://github.com/pkmital/tensorflow_tutorials
        - https://github.com/yaoli/GSN
        - https://github.com/peteykun/GSN
    """
    def __init__(self, num_epochs, batch_size, hidden_dim, n_input, corrupt_prob=0.5, 
                 corrupt_std=0.25, corrupt_type='salt_and_pepper', walkbacks=0, 
                 transfer_fct=tf.nn.sigmoid, W_init_fct=init_xavier, b_init_fct=tf.zeros, 
                 learning_rate=0.001, log_every=None):
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.net_arch = {
            'hidden_dim': hidden_dim,
            'n_input': n_input,
            'n_output': n_input
            }

        self.corrupt_prob = corrupt_prob
        self.corrupt_std = corrupt_std
        self.corrupt_type = corrupt_type
        self.walkbacks = walkbacks

        self.transfer_fct = transfer_fct
        self.W_init_fct = W_init_fct
        self.b_init_fct = b_init_fct
        self.learning_rate = learning_rate
        
        self.log_every = log_every

        # TensorFlow graph input.
        self.x = tf.placeholder(tf.float32, [None, self.net_arch['n_input']])

        # Create autoencoder network.
        self._create_network()
        # Define loss function.
        self._create_loss_optimizer()

        # Initialize the TensorFlow variables.
        init = tf.initialize_all_variables()

        # Launch the session.
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        self.saver = tf.train.Saver(tf.all_variables())

    def _create_network(self):
        """Create a denoising autoencoder network."""
        layer_dim = np.append(np.array(self.net_arch['n_input']), 
            self.net_arch['hidden_dim'])

        self.z, self.y, self.p_X_chain = self._autoencoder(layer_dim=layer_dim)

    def _autoencoder(self, layer_dim):
        """Build a deep denoising autoencoder with tied weights. Implements
        walkback training (optional).

        Parameters
        ----------
        layer_dim : list
            Number of neurons for each layer of the autoencoder.

        Returns
        -------
        z : Tensor
            Inner-most latent representation.
        y : Tensor
            Output reconstruction of the input.
        p_X_chain : array
            Walkback training chain.
        """
        def update_layers(x):
            """Perform layer updates.

            Parameters
            ----------
            x : Tensor
                Input placeholder to the network.

            Returns
            -------
            x : Tensor
                Input placeholder to the network.
            z : Tensor
                Inner-most latent representation.
            y : Tensor
                Output reconstruction of the input.
            """
            if self.corrupt_type == 'gaussian':
                layer_input = corrupt_gaussian(x, std=self.corrupt_std) \
                                * self.corrupt_prob + x * (1 - self.corrupt_prob)
            elif self.corrupt_type == 'salt_and_pepper':
                layer_input = corrupt_salt_and_pepper(x, self.corrupt_prob)
            else:
                layer_input = corrupt_salt_and_pepper(x, self.corrupt_prob)

            # Build the encoder.
            encoder = []
            for layer_i, n_output in enumerate(layer_dim[1:]):
                n_input = int(layer_input.get_shape()[1])
                W = tf.Variable(self.W_init_fct([n_input, n_output]), dtype=tf.float32)
                b = tf.Variable(self.b_init_fct([n_output]), dtype=tf.float32)
                encoder.append(W)
                output = self.transfer_fct(tf.add(tf.matmul(layer_input, W), b))
                layer_input = output

            # Latent representation.
            z = layer_input

            encoder.reverse()
            # Build the decoder using the same weights.
            for layer_i, n_output in enumerate(layer_dim[:-1][::-1]):
                n_input = int(layer_input.get_shape()[1])
                W = tf.transpose(encoder[layer_i])
                b = tf.Variable(self.b_init_fct([n_output]), dtype=tf.float32)
                output = self.transfer_fct(tf.add(tf.matmul(layer_input, W), b))
                layer_input = output

            # Reconstruction through the network.
            y = layer_input

            return (x, y, z)

        # Define p(X|...).
        p_X_chain = []
        # Perform layer updates.
        if self.walkbacks > 0:
            x = self.x
            for i in range(self.walkbacks):
                x, y, z = update_layers(x)
                p_X_chain.append(y)
                x = binomial_vec(y, shape=tf.shape(y)) # sample from p(X|...)
        else:
            x, y, z = update_layers(self.x)

        return (z, y, p_X_chain)

    def _create_loss_optimizer(self):
        """Define the cost function."""
        if self.walkbacks > 0:
            cross_entropies = [binary_crossentropy(y, self.x) for y in self.p_X_chain]
            self.cost = tf.add_n(cross_entropies)
        else:
            self.cost = binary_crossentropy(self.y, self.x)

        # Use ADAM optimizer.
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def reconstruct(self, X):
        """Use DAE to reconstruct given data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data to be reconstructed.

        Returns the reconstructed data.
        """
        return self.sess.run(self.y, feed_dict={self.x: X})

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data to be learned.

        Returns cost of mini-batch.
        """
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def fit(self, X, shuffle=True, display_step=None):
        """Training cycle.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data to be learned.
        """
        if display_step is None:
            display_step = self.log_every
        n_samples = X.shape[0]

        for epoch in range(self.num_epochs):
            if shuffle:
                indices = np.arange(len(X))
                np.random.shuffle(indices)
            avg_cost = 0.
            # Loop over all batches.
            start_idxs = range(0, len(X) - self.batch_size + 1, self.batch_size)
            for start_idx in start_idxs:
                if shuffle:
                    excerpt = indices[start_idx:start_idx + self.batch_size]
                else:
                    excerpt = slice(start_idx, start_idx + self.batch_size)
                batch = np.array(X[excerpt])

                # Fit training using batch data.
                cost = self.partial_fit(batch)
                # Compute average loss.
                avg_cost += cost / n_samples * self.batch_size

            if len(start_idxs) > 0:
                # Display logs per epoch step.
                if display_step and epoch % display_step == 0:
                    print("Epoch: {:d}".format(epoch+1), \
                          "cost: {:.4f}".format(avg_cost))

    def sample(self, in_samples, N):
        """Generate samples via pseudo-Gibbs sampling.

        Parameters
        ----------
        in_samples : ndarray, shape (n_samples, n_features)
            Matrix containing the data from which to sample.
        N : int
            Number of samples to generate.

        Returns samples.
        """
        if not hasattr(in_samples, "__len__"):
            in_samples = [in_samples]

        samples = np.empty(shape=(N, self.net_arch['n_input']))
        for i in range(N):
            if i == 0:
                # Choose a random sample as the initialization.
                in_sample = in_samples[np.random.randint(len(in_samples), size=1)]
                out_sample = self.sess.run(self.y, feed_dict={self.x: in_sample})
            else:
                out_sample = self.sess.run(self.y, feed_dict={
                    self.x: samples[i-1].reshape((1, -1))
                    })
            samples[i] = out_sample

        return samples

    def close(self):
        """Closes the TensorFlow session."""
        self.sess.close()

def main(data, N, args):
    model = DAE(args.num_epochs,
                args.batch_size,
                args.hidden_dim,
                args.n_input,
                args.corrupt_prob,
                args.corrupt_std,
                args.corrupt_type,
                args.walkbacks,
                args.transfer_fct,
                args.W_init_fct,
                args.b_init_fct,
                args.learning_rate,
                args.log_every)
    model.fit(data)
    samples = model.gen_samples(N)
    model.close()
    return samples

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Passes over the training dataset.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Size of minibatches for stochastic optimizers.')
    parser.add_argument('--hidden_dim', type=list, default=(100,),
                        help='Number of units per hidden layer for encoder/decoder.')
    parser.add_argument('--n_input', type=int, default=2,
                        help='Number of inputs to initial layer.')
    parser.add_argument('--corrupt_prob', type=float, default=0.5,
                        help='Probability of generating corrupted values.')
    parser.add_argument('--corrupt_std', type=float, default=0.25,
                        help='Standard deviation of corrupted values (gaussian).')
    parser.add_argument('--corrupt_type', type=str,
                        choices=['gaussian', 'salt_and_pepper'], 
                        default='salt_and_pepper',
                        help='Type of corrupting function.')
    parser.add_argument('--walkbacks', type=int, default=0,
                        help='Number of walkbacks to use.')
    parser.add_argument('--transfer_fct', type=object, default=tf.nn.sigmoid,
                        help='Transfer function for hidden layers.')
    parser.add_argument('--W_init_fct', type=object, default=init_xavier,
                        help='Initialization function for weights.')
    parser.add_argument('--b_init_fct', type=object, default=tf.zeros,
                        help='Initialization function for biases.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate schedule for weight updates.')
    parser.add_argument('--log_every', type=int, default=None,
                        help='Print loss during training after this many steps.')
    return parser.parse_args()

# Test with MNIST.
def test_mnist():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import tensorflow.examples.tutorials.mnist.input_data as input_data

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    n_samples = mnist.train.num_examples

    x_sample_all, _ = mnist.train.next_batch(55000)

    dae = DAE(num_epochs=5,
              batch_size=100,
              hidden_dim=(500, 500),
              n_input=784, # MNIST data input (img shape: 28*28)
              corrupt_prob=0.4,
              corrupt_type='salt_and_pepper',
              walkbacks=0
              )

    dae.fit(x_sample_all, display_step=1)
    x_sample = mnist.test.next_batch(100)[0]
    x_reconstruct = dae.reconstruct(x_sample)

    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2*i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    #plt.show()
    plt.savefig('dae_mnist_rec.png')

    test_input = mnist.test.next_batch(1)[0]
    samples = dae.sample(test_input, 400)
    dae.close()

    fig, ax = plt.subplots(40, 10, figsize=(10, 40))
    for i in range(400):
        ax[i/10][i%10].imshow(np.reshape(samples[i], (28,28)), cmap='gray')
        ax[i/10][i%10].axis('off')
    #plt.show()
    plt.savefig('dae_mnist_samples.png')

if __name__ == '__main__':
    #main(data, 100, parse_args())
    test_mnist()