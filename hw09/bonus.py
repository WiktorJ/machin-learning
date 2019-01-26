import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# %matplotlib inline

loader = np.load("homework_09_data.npz")
train_data = loader['train_data']
train_labels = loader['train_labels']

val_data = loader['val_data']
val_labels = loader['val_labels']

# train_data = np.vstack((train_data, val_data))
# train_labels = np.vstack((train_labels, val_labels))

test_data = loader['test_data']


def batch_data(num_data, batch_size):
    """ Yield batches with indices until epoch is over.

    Parameters
    ----------
    num_data: int
        The number of samples in the dataset.
    batch_size: int
        The batch size used using training.

    Returns
    -------
    batch_ixs: np.array of ints with shape [batch_size,]
        Yields arrays of indices of size of the batch size until the epoch is over.
    """

    data_ixs = np.random.permutation(np.arange(num_data))
    ix = 0
    while ix + batch_size < num_data:
        batch_ixs = data_ixs[ix:ix + batch_size]
        ix += batch_size
        yield batch_ixs


class FeedForwardNet:
    """
    Simple feed forward neural network class
    """

    def __init__(self, hidden_sizes, layer_types, name, l2_reg=0.0):
        """ FeedForwardNet constructor.

        Parameters
        ----------
        hidden_sizes: list of ints
            The sizes of the hidden layers of the network.
        name: str
            The name of the network (used for a VariableScope)
        l2_reg: float
            The strength of L2 regularization (0 means no regularization)
        """

        self.hidden_sizes = hidden_sizes
        self.layer_types = layer_types
        self.name = name
        self.dropout = tf.placeholder_with_default(0.0, shape=(), name="dropout")
        self.l2_reg = l2_reg
        self.weights = []
        self.biases = []

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv1d(self, x, W):
        return tf.nn.conv1d(x, W, strides=1, padding='SAME')

    def max_pool_2x2(self, x):
        return tf.layers.max_pooling1d(x, ksize=2,
                                       strides=1, padding='SAME')

    def conv_layer(self, input, shape):
        W = self.weight_variable(shape)
        b = self.bias_variable([shape[3]])
        return tf.nn.relu(self.conv1d(input, W) + b)

    def build(self, data_dim, num_classes, batch_norm):
        """ Construct the model.

        Parameters
        ----------
        data_dim: int
            The dimensions of the data samples.

        Returns
        -------
        None

        """
        self.X = tf.placeholder(shape=[None, data_dim], dtype=tf.float32, name="data")  # [NxD]
        self.Y = tf.placeholder(shape=[None, num_classes], dtype=tf.float32, name="labels")  # [Nx1]

        with tf.variable_scope(self.name):
            hidden = self.X

            for ix, hidden_size in enumerate(self.hidden_sizes):
                w = tf.Variable(tf.random_normal([hidden.get_shape()[1].value, hidden_size]))
                b = tf.Variable(tf.random_normal([hidden_size]))
                self.weights.append(w)
                self.biases.append(b)
                if batch_norm:
                    hidden = tf.layers.dropout(
                        self.layer_types[ix](tf.layers.batch_normalization(tf.add(tf.matmul(hidden, w), b))),
                        self.dropout)
                else:
                    hidden = tf.layers.dropout(
                        self.layer_types[ix](tf.add(tf.matmul(hidden, w), b)), self.dropout)

            w = tf.Variable(tf.random_normal([hidden.get_shape()[1].value, num_classes]))
            b = tf.Variable(tf.random_normal([num_classes]))
            self.weights.append(w)
            self.biases.append(b)

            self.logits = tf.add(tf.matmul(hidden, w), b)
            self.l2_norm = np.sum([tf.norm(weight) for weight in self.weights])
            self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.logits)
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.logits, 1)), tf.float32))

            self.loss = tf.reduce_sum(self.cross_entropy_loss) + 0.5 * self.l2_reg * self.l2_norm

            self.optimizer = tf.train.AdamOptimizer()
            self.opt_op = self.optimizer.minimize(self.loss, var_list=[*self.weights, *self.biases])

    def build2(self, data_dim, num_classes, batch_norm):
        """ Construct the model.

        Parameters
        ----------
        data_dim: int
            The dimensions of the data samples.

        Returns
        -------
        None

        """
        self.X = tf.placeholder(shape=[None, data_dim], dtype=tf.float32, name="data")  # [NxD]
        self.Y = tf.placeholder(shape=[None, num_classes], dtype=tf.float32, name="labels")  # [Nx1]

        with tf.variable_scope(self.name):
            hidden = self.X
            hidden = \
                tf.layers.dropout(
                    tf.layers.max_pooling1d(
                        tf.layers.batch_normalization(
                            tf.layers.conv1d(tf.reshape(hidden, shape=[-1, 20, 6]), filters=18, kernel_size=2,
                                             strides=1, padding='same')), 2, 1), 0.5)
            hidden = \
                tf.layers.dropout(
                    tf.layers.max_pooling1d(
                        tf.layers.batch_normalization(
                            tf.layers.conv1d(hidden, filters=36, kernel_size=2,
                                         strides=1, padding='same')), 2, 1), 0.5)
            hidden = \
                tf.layers.dropout(
                    tf.layers.max_pooling1d(
                        tf.layers.batch_normalization(
                            tf.layers.conv1d(hidden, filters=72, kernel_size=2,
                                         strides=1, padding='same')), 2, 1), 0.5)
            hidden = \
                tf.layers.dropout(
                    tf.layers.max_pooling1d(
                        tf.layers.batch_normalization(
                            tf.layers.conv1d(hidden, filters=144, kernel_size=2,
                                         strides=1, padding='same')), 2, 1), 0.5)
            hidden = tf.reshape(hidden, [-1, 2304])
            for ix, hidden_size in enumerate(self.hidden_sizes):
                w = tf.Variable(tf.random_normal([hidden.get_shape()[1].value, hidden_size]))
                b = tf.Variable(tf.random_normal([hidden_size]))
                self.weights.append(w)
                self.biases.append(b)
                hidden = tf.layers.dropout(
                    self.layer_types[ix](tf.layers.batch_normalization(tf.add(tf.matmul(hidden, w), b))),
                    self.dropout)

            w = tf.Variable(tf.random_normal([hidden.get_shape()[1].value, num_classes]))
            b = tf.Variable(tf.random_normal([num_classes]))
            self.weights.append(w)
            self.biases.append(b)

            self.logits = tf.add(tf.matmul(hidden, w), b)
            self.l2_norm = sum([tf.norm(weight) for weight in self.weights])
            self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.logits)
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.logits, 1)), tf.float32))

            self.loss = tf.reduce_sum(self.cross_entropy_loss) + 0.5 * self.l2_reg * self.l2_norm

            self.optimizer = tf.train.AdamOptimizer()
            self.opt_op = self.optimizer.minimize(self.loss, var_list=[*self.weights, *self.biases])

    def train(self, train_data, train_labels, val_data, val_labels, epochs=20, dropout=0.0, batch_size=512):
        """ Train the feed forward neural network.

        Parameters
        ----------
        train_data: np.array, dtype float32, shape [N, D]
            The training data. N corresponds to the number of training samples, D to the dimensionality of the data samples/
        train_labels: np.array, shape [N, K]
            The labels of the training data, where K is the number of classes.
        val_data: np.array, dtype float32, shape [N_val, D]
            The validation data. N_val corresponds to the number of validation samples, D to the dimensionality of the data samples/
        val_labels: np.array, shape [N_val, K]
            The labels of the training data, where K is the number of classes.
        epochs: int
            The number of epochs to train for.
        dropout: float
            The dropout rate used during training. 0 corresponds to no dropout.
        batch_size: int
            The batch size used for training.

        Returns
        -------
        None

        """
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        self.session = tf.Session()
        session = self.session

        with session.as_default():
            session.run(tf.global_variables_initializer())

            tr_loss, tr_acc = session.run([self.loss, self.accuracy],
                                          feed_dict={self.X: train_data, self.Y: train_labels})

            val_loss, val_acc = session.run([self.loss, self.accuracy],
                                            feed_dict={self.X: val_data, self.Y: val_labels})

            train_losses.append(tr_loss)
            train_accs.append(tr_acc)

            val_losses.append(val_loss)
            val_accs.append(val_acc)

            for epoch in range(epochs):
                if (epoch + 1) % 25 == 0:
                    print(f"Epoch {epoch+1}/{epochs}")
                for batch_ixs in batch_data(len(train_data), batch_size):
                    _ = session.run(self.opt_op,
                                    feed_dict={self.X: train_data[batch_ixs],
                                               self.Y: train_labels[batch_ixs],
                                               self.dropout: dropout})

                tr_loss, tr_acc = session.run([self.loss, self.accuracy],
                                              feed_dict={self.X: train_data, self.Y: train_labels})

                val_loss, val_acc = session.run([self.loss, self.accuracy],
                                                feed_dict={self.X: val_data, self.Y: val_labels})

                train_losses.append(tr_loss)
                train_accs.append(tr_acc)

                val_losses.append(val_loss)
                val_accs.append(val_acc)

        self.hist = {'train_loss': np.array(train_losses),
                     'train_accuracy': np.array(train_accs),
                     'val_loss': np.array(val_losses),
                     'val_accuracy': np.array(val_accs)}


layers = [
    # ([100, 100], [tf.nn.relu, tf.nn.relu]),
    # ([100, 100, 100], [tf.nn.relu, tf.nn.relu,tf.nn.relu]),
    # ([100, 100, 100, 100], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([100, 100, 100, 100, 100], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([100, 100, 100, 100, 100, 100], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([50, 50], [tf.nn.relu, tf.nn.relu]),
    # ([50, 50, 50], [tf.nn.relu, tf.nn.relu,tf.nn.relu]),
    # ([50, 50, 50, 50], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([50, 50, 50, 50, 50], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([50, 50, 50, 50, 50, 50], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([200, 200], [tf.nn.relu, tf.nn.relu]),
    ([400, 400], [tf.nn.relu, tf.nn.relu]),
    # ([200, 200], [tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([400, 400, 400], [tf.nn.relu, tf.nn.relu,tf.nn.relu]),
    # ([200, 200, 200, 200], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([500, 500, 500, 500], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([600, 600, 600, 600], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([700, 700, 700, 700], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([500, 400, 300, 200], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([400, 400, 400, 400], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([600, 600, 600, 600, 600], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([200, 200, 200, 200, 200], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([300, 300, 300, 300, 300], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([350, 350, 350, 350, 350, 350], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([100, 100, 100, 100, 100, 100], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
    # ([200, 200, 200, 200, 200, 200], [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]),
]
epochs = 100
batch_size = 512

# batch_norm = [False]
# dropouts = [0]
# l2 = [0]

batch_norm = [True]
dropouts = [0.5]
l2 = [0.0001]

# dropouts = [0, 0.5]
# l2 = [5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
#
# batch_norm = [True, False]
# dropouts = [0, 0.5]
# l2 = [0, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]

res = []
print(f"EPOCHS: {epochs}")
for d in dropouts:
    for l in l2:
        for bn in batch_norm:
            for layer in layers:
                model = FeedForwardNet(layer[0], layer[1],
                                       f"model{d}{l}{bn}{np.random.random()}",
                                       l2_reg=l)
                model.build(train_data.shape[1], num_classes=train_labels.shape[1], batch_norm=bn)
                model.train(train_data, train_labels, val_data, val_labels, epochs,
                            batch_size=batch_size, dropout=d)

                train_acc = model.hist['train_accuracy'][-1]
                val_acc = model.hist['val_accuracy'][-1]
                res.append({
                    "dropout": d,
                    "l2": l,
                    "bn": bn,
                    "sizes": layer[0],
                    "train_acc": train_acc,
                    "val_acc": val_acc
                })
                print(f"Dropout: {d}, l2: {l}, Batch norm: {bn}, Sizes: {layer[0]}")
                print(f"Training accuracy: {train_acc:.3f}")
                print(f"Validation accuracy: {val_acc:.3f}")
                print("-----------------------")

                # test_preds = model.logits.eval({model.X: test_data},
                #                                session=model.session).argmax(1)
                #
                # np.savetxt("bonus_solution2.txt", test_preds, fmt='%i')

res = sorted(res, key=lambda k: k['val_acc'], reverse=True)
print("SORTED RES")
print(res, sep="\n")
