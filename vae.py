import argparse
import numbers

import numpy as np
import tensorflow as tf


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None or int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Notes
    -----
    This routine is from scikit-learn. See:
    http://scikit-learn.org/stable/developers/utilities.html#validation-tools.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState"
        " instance" % seed
    )


def init_xavier(fan, constant=1):
    """Xavier initialization of network weights."""
    fan_in, fan_out = fan[0], fan[1]
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform(
        (fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32
    )


def binary_crossentropy(output, target, offset=1e-10):
    """Compute the binary cross-entropy per sample.

    Add offset to avoid evaluation of log(0.0).
    """
    output_ = tf.clip_by_value(output, offset, 1 - offset)
    return -tf.reduce_sum(
        target * tf.log(output_) + (1 - target) * tf.log(1 - output_), 1
    )


class VAE(object):
    """Variational Autoencoder (VAE) implemented using TensorFlow.

    The VAE learns a directed probabilistic model of input data represented by
    continuous latent variables, using a neural network to perform efficient
    approximate posterior inference [1].

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions realized by multilayer perceptrons. The VAE can be learned
    end-to-end.

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

    n_z : int
        Number of units in the latent layer.

    transfer_fct : object
        Transfer function for hidden layers.

    W_init_fct : object
        Initialization function for weights.

    b_init_fct : object
        Initialization function for biases.

    learning_rate : float
        Learning rate schedule for weight updates.

    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.

    log_every : int
        Print loss after this many steps.

    References
    ----------
    .. [1] D. P. Kingma and M. Welling. "Auto-Encoding Variational Bayes".
           arXiv preprint arXiv:1312.6114, 2013.

    Notes
    -----
    Based on related code:
        - https://jmetzen.github.io/2015-11-27/vae.html
    """

    def __init__(
        self,
        num_epochs,
        batch_size,
        hidden_dim,
        n_input,
        n_z,
        transfer_fct=tf.nn.sigmoid,
        W_init_fct=init_xavier,
        b_init_fct=tf.zeros,
        learning_rate=0.001,
        random_state=None,
        log_every=None,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.net_arch = {
            "hidden_dim": hidden_dim,
            "n_z": n_z,
            "n_input": n_input,
            "n_output": n_input,
        }

        self.transfer_fct = transfer_fct
        self.W_init_fct = W_init_fct
        self.b_init_fct = b_init_fct

        self.learning_rate = learning_rate

        self.random_state = check_random_state(random_state)
        tf.set_random_seed(random_state)

        self.log_every = log_every

        # TensorFlow graph input.
        self.x = tf.placeholder(tf.float32, [None, self.net_arch["n_input"]])

        # Create autoencoder network.
        self._create_network()
        # Define the loss function.
        self._create_loss_optimizer()

        # Initialize the TensorFlow variables.
        init = tf.global_variables_initializer()

        # Launch the session.
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        self.saver = tf.train.Saver(tf.global_variables())

    def _create_network(self):
        """Initialize the autoencoder network weights and biases."""
        layer_dim = np.append(
            np.array(self.net_arch["n_input"]), self.net_arch["hidden_dim"]
        )

        # Use recognition network to determine mean and (log) variance of
        # Gaussian distribution in latent space.
        self.z_mean, self.z_log_sigma_sq = self._recognition_network(
            self.x, layer_dim
        )

        # Use the reparameterization trick to draw a sample, z, from the
        # Gaussian distribution, with epsilon as an auxiliary noise variable.

        eps = tf.random_normal(
            (self.batch_size, self.net_arch["n_z"]), 0, 1, dtype=tf.float32
        )
        # z = mu + sigma * epsilon
        self.z = tf.add(
            self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps)
        )

        # Use generator to determine mean of Bernoulli distribution of
        # reconstructed input.
        self.x_reconstr_mean = self._generator_network(self.z, layer_dim)

    def _recognition_network(self, layer_input, layer_dim):
        """Define the recognition network.

        The probabilistic encoder (recognition network) maps inputs onto a
        normal distribution in latent space. The transformation is
        parameterized and can be learned.

        Parameters
        ----------
        layer_dim : list
            Number of neurons for each layer of the recognition network.

        Returns
        -------
        z_mean : Tensor
            Mean of the latent space.
        z_log_sigma_sq : Tensor
            Log sigma squared of the latent space.
        """
        for layer_i, n_output in enumerate(layer_dim[1:]):
            n_input = int(layer_input.get_shape()[1])
            W = tf.Variable(
                self.W_init_fct([n_input, n_output]), dtype=tf.float32
            )
            b = tf.Variable(self.b_init_fct([n_output]), dtype=tf.float32)
            output = self.transfer_fct(tf.add(tf.matmul(layer_input, W), b))
            layer_input = output

        n_dims = self.net_arch["hidden_dim"][-1]

        W_out_mean = tf.Variable(
            self.W_init_fct([n_dims, self.net_arch["n_z"]])
        )
        W_out_log_sigma = tf.Variable(
            self.W_init_fct([n_dims, self.net_arch["n_z"]])
        )
        b_out_mean = tf.Variable(
            self.b_init_fct([self.net_arch["n_z"]], dtype=tf.float32)
        )
        b_out_log_sigma = tf.Variable(
            self.b_init_fct([self.net_arch["n_z"]], dtype=tf.float32)
        )

        z_mean = tf.add(tf.matmul(layer_input, W_out_mean), b_out_mean)
        z_log_sigma_sq = tf.add(
            tf.matmul(layer_input, W_out_log_sigma), b_out_log_sigma
        )
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, layer_input, layer_dim):
        """Define the generator network.

        The probabilistic decoder (decoder network) maps points in latent
        space onto a Bernoulli distribution in data space. The transformation
        is parameterized and can be learned.

        Parameters
        ----------
        layer_dim : list
            Number of neurons for each layer of the generator network.

        Returns
        -------
        x_reconstr_mean : Tensor
            Mean of the reconstructed data.
        """
        for layer_i, n_output in enumerate(reversed(layer_dim[1:])):
            n_input = int(layer_input.get_shape()[1])
            W = tf.Variable(
                self.W_init_fct([n_input, n_output]), dtype=tf.float32
            )
            b = tf.Variable(self.b_init_fct([n_output]), dtype=tf.float32)
            output = self.transfer_fct(tf.add(tf.matmul(layer_input, W), b))
            layer_input = output

        n_dims = self.net_arch["hidden_dim"][0]

        W_out_mean = tf.Variable(
            self.W_init_fct([n_dims, self.net_arch["n_output"]])
        )
        b_out_mean = tf.Variable(
            self.b_init_fct([self.net_arch["n_output"]], dtype=tf.float32)
        )

        x_reconstr_mean = tf.nn.sigmoid(
            tf.add(tf.matmul(layer_input, W_out_mean), b_out_mean)
        )
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        """Define the cost function.

        The loss is composed of two terms:
        1.) The reconstruction loss (the negative log probability of the
            input under the reconstructed Bernoulli distribution induced by
            the decoder in the data space). This can be interpreted as the
            number of "nats" required for reconstructing the input when the
            activation in latent space is given.

        2.) The latent loss (the Kullback-Leibler divergence between the
            distribution in latent space induced by the encoder on the data
            and some prior). This acts as a kind of regularizer, and can be
            interpreted as the number of "nats" required for transmitting the
            latent space distribution given the prior.
        """
        reconstr_loss = binary_crossentropy(self.x_reconstr_mean, self.x)
        latent_loss = -0.5 * tf.reduce_mean(1 + self.z_log_sigma_sq
                                            - tf.square(self.z_mean)
                                            - tf.exp(self.z_log_sigma_sq), 1)
        # Average over batch.
        self.cost = tf.reduce_mean(tf.add(reconstr_loss, latent_loss))
        # Use ADAM optimizer.
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.opt = opt.minimize(self.cost)

    def transform(self, X):
        """Transform data by mapping it into the latent space.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data to be transformed.

        Note: This maps to mean of the distribution; we could alternatively
        sample from the Gaussian distribution.
        """
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is generated.
        Otherwise, z_mu is drawn from prior in latent space.

        Note: This maps to mean of the distribution; we could alternatively
        sample from the Gaussian distribution.
        """
        if z_mu is None:
            z_mu = self.random_state.normal(
                size=(self.batch_size, self.net_arch["n_z"])
            )
            return self.sess.run(
                self.x_reconstr_mean, feed_dict={self.z: z_mu}
            )
        else:
            z_mu = np.reshape(z_mu, (1, self.net_arch["n_z"]))
            z = np.repeat(z_mu, self.batch_size, axis=0)
            return self.sess.run(self.x_reconstr_mean, feed_dict={self.z: z})

    def reconstruct(self, X):
        """Use VAE to reconstruct given data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data to be reconstructed.

        Returns the reconstructed data.
        """
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.x: X})

    def sample(self, n_samples):
        """Generate samples.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns samples.
        """
        samples = np.empty(shape=(n_samples, self.net_arch["n_input"]))
        for i in range(n_samples):
            # Note: The dimensionality of z_mu is fixed, so we cannot generate
            # `n_samples` samples directly. Instead, we can take the first
            # sample or a random sample and repeat. Alternatively, we could
            # save the graph variables and reinitialize the graph with z_mu of
            # size `n_samples`.
            #samples[i] = self.generate()[0]
            samples[i] = self.generate()[
                self.random_state.randint(self.batch_size, size=1)
            ]
        return samples

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data to be learned.

        Returns cost of mini-batch.
        """
        opt, cost = self.sess.run(
            (self.opt, self.cost), feed_dict={self.x: X}
        )
        return cost

    def fit(self, X, shuffle=True, display_step=5):
        """Training cycle.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data to be learned.

        Returns
        -------
        self : object
            Returns self.
        """
        if display_step is None:
            display_step = self.log_every
        n_samples = X.shape[0]

        for epoch in range(self.num_epochs):
            if shuffle:
                indices = np.arange(len(X))
                self.random_state.shuffle(indices)
            avg_cost = 0.
            # Loop over all batches.
            start_idxs = range(
                0, len(X) - self.batch_size + 1, self.batch_size
            )
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
                    print(
                        "Epoch: {:d}".format(epoch + 1),
                        "cost: {:.4f}".format(avg_cost)
                    )

        return self

    def close(self):
        """Closes the TensorFlow session."""
        self.sess.close()


def main(data, n_samples, args):
    model = VAE(
        args.num_epochs,
        args.batch_size,
        args.hidden_dim,
        args.n_input,
        args.n_z,
        args.transfer_fct,
        args.W_init_fct,
        args.b_init_fct,
        args.learning_rate,
        args.random_state,
        args.log_every,
    )
    model.fit(data)
    samples = model.sample(n_samples)
    model.close()
    return samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Passes over the training dataset.")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Size of minibatches for stochastic optimizers.")
    parser.add_argument("--hidden_dim", type=list, default=(100,),
                        help="Number of units per hidden layer for "
                             "encoder/decoder.")
    parser.add_argument("--n_input", type=int, default=2,
                        help="Number of inputs to initial layer.")
    parser.add_argument("--n_z", type=int, default=2,
                        help="Number of units in the latent layer.")
    parser.add_argument("--transfer_fct", type=object, default=tf.nn.sigmoid,
                        help="Transfer function for hidden layers.")
    parser.add_argument("--W_init_fct", type=object, default=init_xavier,
                        help="Initialization function for weights.")
    parser.add_argument("--b_init_fct", type=object, default=tf.zeros,
                        help="Initialization function for biases.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate schedule for weight updates.")
    parser.add_argument("--random_state", type=int, default=None,
                        help="The seed used by the random number generator.")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Print loss after this many steps.")
    return parser.parse_args()


# Test with MNIST.
def test_mnist():
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mnist = tf.keras.datasets.mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    img_rows, img_cols = 28, 28
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    X_train = X_train.reshape((n_train, img_rows*img_cols))
    X_test = X_test.reshape((n_test, img_rows*img_cols))

    # Standardize.
    X_train = X_train / 256.
    X_test = X_test / 256.

    # One-hot encode.
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    vae = VAE(
        num_epochs=10,
        batch_size=100,
        hidden_dim=(512, 256),
        n_input=784,  # MNIST data input (img shape: 28*28)
        n_z=64,  # dimensionality of latent space
    )
    vae.fit(X_train, display_step=1)

    X_test_samples = X_test[:100]
    X_reconstruct = vae.reconstruct(X_test_samples)
    vae.close()

    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2*i+1)
        plt.imshow(X_test_samples[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2*i+2)
        plt.imshow(X_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    #plt.show()
    plt.savefig("vae_mnist_rec.png")

    vae_2d = VAE(
        num_epochs=10,
        batch_size=100,
        hidden_dim=(512, 256),
        n_input=784,  # MNIST data input (img shape: 28*28)
        n_z=2,  # dimensionality of latent space
    )

    vae_2d.fit(X_train, display_step=1)
    X_test_samples, y_test_samples = X_test[:5000], y_test[:5000]
    z_mu = vae_2d.transform(X_test_samples)

    plt.figure(figsize=(8, 6))
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_test_samples, 1))
    plt.colorbar()
    #plt.show()
    plt.savefig("vae_2d_mnist_zspace.png")

    nx = ny = 20
    X_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28 * ny, 28 * nx))
    for i, yi in enumerate(X_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]])
            X_mean = vae_2d.generate(z_mu)
            X_mean = X_mean[0].reshape(28, 28)
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = X_mean

    plt.figure(figsize=(8, 10))
    _, _ = np.meshgrid(X_values, y_values)
    plt.imshow(canvas, origin="upper")
    plt.tight_layout()
    #plt.show()
    plt.savefig("vae_2d_mnist_zspace_samples.png")

    samples = vae_2d.sample(400)
    vae_2d.close()

    fig, ax = plt.subplots(40, 10, figsize=(10, 40))
    for i in range(400):
        ax[i/10][i%10].imshow(np.reshape(samples[i], (28, 28)), cmap="gray")
        ax[i/10][i%10].axis("off")
    #plt.show()
    plt.savefig("vae_2d_mnist_samples.png")


if __name__ == "__main__":
    #main(data, 100, parse_args())
    test_mnist()
