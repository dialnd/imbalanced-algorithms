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


def binary_crossentropy(preds, targets, offset=1e-10, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = preds`, `z = targets`. The logistic loss is
        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))
    """
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(
            -(targets * tf.log(preds + offset) +
                (1. - targets) * tf.log(1. - preds + offset))
        )


def lrelu(X, leak=0.2, name="lrelu"):
    """Leaky rectified linear unit (LReLU)."""
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * X + f2 * abs(X)


def linear(
    input_,
    output_size,
    scope=None,
    stddev=0.5,
    bias_start=0.0,
    with_w=False,
):
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
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(
            "Matrix",
            [shape[1], output_size],
            tf.float32,
            tf.random_normal_initializer(stddev=stddev),
        )
        bias = tf.get_variable(
            "bias",
            [output_size],
            initializer=tf.constant_initializer(bias_start),
        )
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def optimizer(loss, var_list):
    """Pre-training optimizer."""
    initial_learning_rate = 0.02
    decay = 0.95
    num_decay_steps = 150
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True,
    )
    learning_rate = initial_learning_rate
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list,
    )
    return optimizer


class GeneratorDistribution(object):
    """Random noise generator."""

    def __init__(self, n_input, random_state=None):
        self.n_input = n_input
        self.random_state = random_state

    def sample(self, N):
        s = np.empty([N, self.n_input])
        for i in range(self.n_input):
            s[:, i] = self.random_state.uniform(low=0.0, high=1.0, size=N)
        return s


class GAN(object):
    """Generative Adversarial Network (GAN) implemented using TensorFlow.

    The GAN framework uses two iteratively trained adversarial networks to
    estimate a generative process. A generative model, G, captures the data
    distribution, while a discriminative model, D, estimates the probability
    that a sample came from the training data rather than from G, the
    generative model [1].

    Parameters
    ----------
    num_epochs : int
        Passes over the training dataset.

    batch_size : int
        Size of minibatches for stochastic optimizers.

    d_hidden_dim : list
        Discriminator number of units per hidden layer.

    g_hidden_dim : list
        Generator number of units per hidden layer.

    n_input : int
        Number of inputs to initial layer.

    stddev : float
        The standard deviation for the initialization noise.

    pretrain : bool
        Use unsupervised pre-training to initialize the discriminator weights.

    d_transfer_fct : object
        Discriminator transfer function for hidden layers.

    g_transfer_fct : object
        Generator transfer function for hidden layers.

    W_init_fct : object
        Initialization function for weights.

    b_init_fct : object
        Initialization function for biases.

    d_learning_rate : float
        Discriminator learning rate schedule for weight updates.

    g_learning_rate : float
        Generator learning rate schedule for weight updates.

    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.

    log_every : int
        Print loss after this many steps.

    References
    ----------
    .. [1] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D.
           Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. "Generative
           Adversarial Nets". Advances in Neural Information Processing
           Systems 27 (NIPS), 2014.

    Notes
    -----
    Based on related code:
        - https://github.com/AYLIEN/gan-intro
        - https://github.com/ProofByConstruction/better-explanations
    """

    def __init__(
        self,
        num_epochs,
        batch_size,
        d_hidden_dim,
        g_hidden_dim,
        n_input,
        stddev=0.5,
        pretrain=False,
        d_transfer_fct=lrelu,
        g_transfer_fct=tf.nn.relu,
        W_init_fct=init_xavier,
        b_init_fct=tf.zeros,
        d_learning_rate=0.01,
        g_learning_rate=0.0005,
        random_state=None,
        log_every=None,
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.net_arch = {
            "d_hidden_dim": d_hidden_dim,
            "g_hidden_dim": g_hidden_dim,
            "n_input": n_input,
            "n_output": n_input,
        }

        self.stddev = stddev

        self.d_pretrain = pretrain

        self.d_transfer_fct = d_transfer_fct
        self.g_transfer_fct = g_transfer_fct
        self.W_init_fct = W_init_fct
        self.b_init_fct = b_init_fct

        self.d_learning_rate = d_learning_rate
        self.g_learning_rate = g_learning_rate

        self.random_state = check_random_state(random_state)
        tf.set_random_seed(random_state)

        self.log_every = log_every

        # Initialize generator distribution.
        self.gen = GeneratorDistribution(
            self.net_arch["n_input"], self.random_state)

        # Create discriminator and generator networks.
        self._create_networks()
        # Define the loss function.
        self._create_loss_optimizer()

        # Initialize the TensorFlow variables.
        init = tf.global_variables_initializer()

        # Launch the session.
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        self.saver = tf.train.Saver(tf.global_variables())

    def _generator(
        self,
        layer_input,
        layer_dim,
        output_dim,
        batch_norm=True,
        stddev=0.5,
    ):
        """Define the generator network.

        Parameters
        ----------
        layer_input : Tensor
            Input to the initial layer.

        layer_dim : list
            Number of neurons for each hidden layer of the generator network.

        output_dim : int
            Number of neurons for the output of the generator network.

        Returns the output of the generator network.
        """
        for layer_i, n_output in enumerate(layer_dim):
            if batch_norm:
                layer_input = tf.contrib.layers.batch_norm(
                    layer_input, scope="G_{0}".format(layer_i))
            output = self.g_transfer_fct(
                linear(
                    layer_input,
                    n_output,
                    scope="G_{0}".format(layer_i),
                    stddev=stddev
                )
            )
            layer_input = output
        return tf.nn.tanh(
            linear(layer_input, output_dim, scope="G_final", stddev=stddev)
        )
        #return tf.nn.relu(
        #    linear(layer_input, output_dim, scope="G_final", stddev=stddev)
        #)

    def _discriminator(self, layer_input, layer_dim, stddev=0.5):
        """Define the discriminator network.

        Parameters
        ----------
        layer_input : Tensor
            Input to the initial layer.

        layer_dim : list
            Number of neurons for each hidden layer of the discriminator network.

        Returns the output of the discriminator network. The output layer has
        one neuron for binary discrimination.
        """
        for layer_i, n_output in enumerate(layer_dim):
            output = self.d_transfer_fct(
                linear(
                    layer_input,
                    n_output,
                    scope="D_{0}".format(layer_i),
                    stddev=stddev,
                )
            )
            layer_input = output
        # return tf.nn.sigmoid(linear(layer_input, 1, scope="D_final"))
        return lrelu(linear(layer_input, 1, scope="D_final", stddev=stddev))

    def _create_networks(self):
        """Initialize the discriminator and generator networks.

        In order to make sure that the discriminator is providing useful gradient
        information to the generator from the start, we can pretrain the
        discriminator using a maximum likelihood objective. We define the network
        for this pretraining step scoped as D_pre.
        """
        # Pretrain (optional).
        if self.d_pretrain:
            with tf.variable_scope("D_pre"):
                self.pre_input = tf.placeholder(
                    tf.float32, shape=(self.batch_size, self.net_arch["n_input"])
                )
                self.pre_labels = tf.placeholder(
                    tf.float32, shape=(self.batch_size, self.net_arch["n_input"])
                )
                D_pre = self._discriminator(
                    self.pre_input,
                    self.net_arch["d_hidden_dim"],
                    stddev=self.stddev
                )
                self.pre_loss = tf.reduce_mean(
                    tf.square(D_pre - self.pre_labels))
                self.pre_opt = optimizer(self.pre_loss, None)

        # Define the generator network.
        with tf.variable_scope("G"):
            self.z = tf.placeholder(
                tf.float32, [None, self.net_arch["n_input"]], name="z"
            )
            self.G = self._generator(
                self.z,
                self.net_arch["g_hidden_dim"],
                self.net_arch["n_output"],
                batch_norm=True,
                stddev=self.stddev,
            )

        # Define the discriminator network.
        with tf.variable_scope("D") as scope:
            self.x = tf.placeholder(
                tf.float32, [None, self.net_arch["n_output"]], name="x"
            )
            self.D1 = self._discriminator(
                self.x, self.net_arch["d_hidden_dim"], stddev=self.stddev
            )
            scope.reuse_variables()
            self.D2 = self._discriminator(
                self.G, self.net_arch["d_hidden_dim"], stddev=self.stddev
            )

    def _create_loss_optimizer(self):
        """Define the cost functions."""
        # Define two discriminator losses, based on the fake and real
        # discriminator predictions.
        self.loss_d_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.D1), logits=self.D1
            )
        )
        self.loss_d_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self.D2), logits=self.D2
            )
        )

        # Define the loss for the discriminator and generator networks.
        self.loss_d = tf.add(self.loss_d_real, self.loss_d_fake)
        self.loss_g = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.D2), logits=self.D2
            )
        )

        t_vars = tf.trainable_variables()
        if self.d_pretrain:
            self.d_pre_vars = [
            var for var in t_vars if var.name.startswith("D_pre/")
        ]
        self.d_vars = [var for var in t_vars if var.name.startswith("D/")]
        self.g_vars = [var for var in t_vars if var.name.startswith("G/")]

        # Define optimizers for the discriminator and generator networks.
        opt_d = tf.train.AdamOptimizer(self.d_learning_rate, beta1=0.5)
        opt_g = tf.train.AdamOptimizer(self.g_learning_rate, beta1=0.5)
        self.opt_d = opt_d.minimize(self.loss_d, var_list=self.d_vars)
        self.opt_g = opt_g.minimize(self.loss_g, var_list=self.g_vars)

    def sample(self, n_samples):
        """Generate samples.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns samples.
        """
        batch_size = min(self.batch_size, n_samples)

        # Generate samples.
        zs = np.empty([n_samples, self.net_arch["n_input"]])
        for i in range(self.net_arch["n_input"]):
            zs[:, i] = self.random_state.uniform(
                low=0.0, high=1.0, size=n_samples
            )

        samples = np.zeros((n_samples, self.z.get_shape()[1]))
        for i in range(n_samples // batch_size):
            z_batch = np.reshape(
                zs[batch_size * i:batch_size * (i+1)],
                (batch_size, self.z.get_shape()[1])
            )
            samples[batch_size * i:batch_size * (i+1)] = self.sess.run(
                self.G, feed_dict={self.z: z_batch}
            )

        return samples

    def partial_fit(self, X, Z):
        """Train model based on mini-batch of input data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data to be learned for the discriminator.

        Z : ndarray, shape (n_samples, n_features)
            Matrix containing the data to be learned for the generator.

        Returns cost of mini-batch.
        """
        # Update discriminator.
        opt_d, cost_d = self.sess.run(
            (self.opt_d, self.loss_d), feed_dict={self.z: Z, self.x: X}
        )
        # Update generator.
        opt_g, cost_g = self.sess.run(
            (self.opt_g, self.loss_g), feed_dict={self.z: Z}
        )

        return (cost_d, cost_g)

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

        # Pretrain the discriminator.
        if self.d_pretrain:
            num_pretrain_steps = 1000
            for step in xrange(num_pretrain_steps):
                d = np.empty([self.batch_size, self.net_arch["n_input"]])
                for i in range(self.net_arch["n_input"]):
                    d[:, i] = (self.random_state.random_sample(
                        self.batch_size) - 0.5
                    ) * 10.0
                labels = np.empty([self.batch_size, self.net_arch["n_input"]])
                for i in range(self.net_arch["n_input"]):
                    labels[:, i] = self.random_state.uniform(
                        low=0.0, high=1.0, size=d.shape[0]
                    )
                pretrain_loss, _ = self.sess.run([self.pre_loss, self.pre_opt], {
                    self.pre_input: np.reshape(d, (self.batch_size, -1)),
                    self.pre_labels: np.reshape(
                        labels, (self.batch_size, self.net_arch["n_input"])
                    ),
                })
            self.weightsD = self.sess.run(self.d_pre_vars)
            # Copy the weights from pre-training over to the new discriminator
            # network.
            for i, v in enumerate(self.d_vars):
                self.sess.run(v.assign(self.weightsD[i]))

        for epoch in range(self.num_epochs):
            if shuffle:
                indices = np.arange(len(X))
                self.random_state.shuffle(indices)
            # Loop over all batches.
            start_idxs = range(
                0, len(X) - self.batch_size + 1, self.batch_size)
            for start_idx in start_idxs:
                if shuffle:
                    excerpt = indices[start_idx:start_idx + self.batch_size]
                else:
                    excerpt = slice(start_idx, start_idx + self.batch_size)
                batch_x = np.array(X[excerpt])
                batch_z = self.gen.sample(self.batch_size)

                # Fit training using batch data.
                cost_d, cost_g = self.partial_fit(batch_x, batch_z)

            if len(start_idxs) > 0:
                if display_step and epoch % display_step == 0:
                    print(
                        "Epoch: {:d}".format(epoch + 1),
                        "loss_d_real: {:.4f}".format(
                            self.loss_d_real.eval({self.x: batch_x})
                        ),
                        "loss_d_fake: {:.4f}".format(
                            self.loss_d_fake.eval({self.z: batch_z})
                        ),
                        "loss_g: {:.4f}".format(
                            self.loss_g.eval({self.z: batch_z}))
                    )

        return self

    def close(self):
        """Closes the TensorFlow session."""
        self.sess.close()


def main(data, n_samples, args):
    tf.reset_default_graph()
    model = GAN(
        args.num_epochs,
        args.batch_size,
        args.d_hidden_dim,
        args.g_hidden_dim,
        args.n_input,
        args.stddev,
        args.pretrain,
        args.d_transfer_fct,
        args.g_transfer_fct,
        args.W_init_fct,
        args.b_init_fct,
        args.d_learning_rate,
        args.g_learning_rate,
        args.random_state,
        args.log_every,
    )
    model.fit(data)
    samples = model.sample(n_samples)
    model.close()
    return samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=1000,
                        help="Passes over the training dataset.")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Size of minibatches for stochastic optimizers.")
    parser.add_argument("--n_d_hidden_dim", type=list, default=(100,),
                        help="Discriminator number of units per hidden layer.")
    parser.add_argument("--n_g_hidden_dim", type=list, default=(100,),
                        help="Generator number of units per hidden layer.")
    parser.add_argument("--n_input", type=int, default=2,
                        help="Number of inputs to the initial layer.")
    parser.add_argument("--stddev", type=int, default=0.5,
                        help="The standard deviation for the initialization "
                             "noise.")
    parser.add_argument("--pretrain", type=int, default=False,
                        help="Use unsupervised pre-training to initialize the "
                        "discriminator weights.")
    parser.add_argument("--d_transfer_fct", type=object, default=lrelu,
                        help="Discriminator transfer function for hidden "
                             "layers.")
    parser.add_argument("--g_transfer_fct", type=object, default=tf.nn.relu,
                        help="Generator transfer function for hidden layers.")
    parser.add_argument("--W_init_fct", type=object, default=init_xavier,
                        help="Initialization function for weights.")
    parser.add_argument("--b_init_fct", type=object, default=tf.zeros,
                        help="Initialization function for biases.")
    parser.add_argument("--d_learning_rate", type=float, default=0.01,
                        help="Discriminator learning rate schedule for weight "
                             "updates.")
    parser.add_argument("--g_learning_rate", type=float, default=0.0005,
                        help="Generator learning rate schedule for weight "
                             "updates.")
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

    gan = GAN(
        num_epochs=10,
        batch_size=100,
        d_hidden_dim=(512, 256),
        g_hidden_dim=(512, 256, 64),
        n_input=784,  # MNIST data input (img shape: 28*28)
        stddev=0.01,  # standard deviation for initialization noise
        pretrain=False,
    )

    gan.fit(X_train, display_step=1)
    samples = gan.sample(400)
    gan.close()

    fig, ax = plt.subplots(40, 10, figsize=(10, 40))
    for i in range(400):
        ax[i/10][i%10].imshow(np.reshape(samples[i], (28, 28)), cmap="gray")
        ax[i/10][i%10].axis("off")
    #plt.show()
    plt.savefig("gan_mnist_samples.png")


if __name__ == "__main__":
    #main(data, 100, parse_args())
    test_mnist()
