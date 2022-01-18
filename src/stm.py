# -*- coding: utf-8 -*-

"""
Guided Topological Map

An organizing mav gtmp whose topology can be guided by a teaching signal.
Generalizes SOMs.

"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.engine import base_layer_utils


def is_2d(func):
    func.type = "2d"
    return func


def is_1d(func):
    func.type = "1d"
    return func


@is_1d
def radial(mean, sigma, size):
    """Gives radial bases on a 2D space, given a list of means and a std_dev.

    Args:
        mean (list): vector of means of radians
        sigma (float): standard deviation of radiants

    Returns:
        (np.array): a radial basis of the input
    """

    x = tf.range(size, dtype="float")
    diff = tf.reshape(x, [1, -1]) - tf.reshape(mean, [-1, 1])
    radial_basis = tf.exp(-0.5 * tf.pow(sigma, -2) * tf.pow(diff, 2))
    return radial_basis


@is_2d
def radial2d(mean, sigma, size):
    """Gives radial bases on a 2D space, flattened into 1d vectors,
    given a list of means and a std_dev.

    Es.

    size: 64              ┌────────┐         ┌─────────────────────────────────────────────────────────────────┐
    mean: (5, 4)  ----->  │........│  -----> │ ....................,+,....,oOo,...+O@O+...,oOo,....,+,.........│
    sigma: 1.5            │........│         └─────────────────────────────────────────────────────────────────┘
                          │....,+,.│
                          │...,oOo,│
                          │...+O@O+│
                          │...,oOo,│
                          │....,+,.│
                          │........│
                          └────────┘

    Args:
        mean (list): vector of means of radians
        sigma (float): standard deviation of radiants
        size (int): dimension of the flattened gaussian (side is sqrt(dim))

    Returns:
        (np.array): each row is a flattened radial basis in
                 the (sqrt(dim), sqrt(dim)) space

    """
    side = tf.sqrt(tf.cast(size, dtype="float"))
    grid_points = make_grid(side)
    diff = tf.expand_dims(grid_points, axis=0) - tf.expand_dims(mean, axis=1)
    radial_basis = tf.exp(-0.5 * tf.pow(sigma, -2) * tf.pow(tf.norm(diff, axis=-1), 2))

    return radial_basis


def make_grid(side):
    x = tf.range(side, dtype="float")
    Y, X = tf.meshgrid(x, x)
    grid_points = tf.transpose(tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])]))
    return grid_points


class STM(keras.layers.Layer):
    """A generic topological map"""

    def __init__(
        self,
        output_size,
        sigma,
        learn_intrinsic_distances=True,
        extrinsic_distances=None,
        radial_fun=radial2d,
        **kwargs
    ):
        """
        Args:
            output_size (int): number of elements in the output layer (shape will be
                                (sqrt(output_size), sqrt(output_size)))
            sigma (float): starting value of the extension of the learning neighbourhood
                           in the output layer.
            learn_intrinsic_distances (boolean): if learning depends on the distance of prototypes from inputs.
            exrinsic_distances (Tensor): if learning depends on the distance of prototypes from targets.

        """

        self.output_size = output_size
        self._sigma = sigma
        self.learn_intrinsic_distances = learn_intrinsic_distances
        self.extrinsic_distances = extrinsic_distances
        self.radial_fun = radial_fun
        self.side = tf.sqrt(tf.cast(output_size, dtype="float"))
        self.grid = make_grid(self.side)
        self.grid = tf.expand_dims(self.grid, axis=(0))

        super(STM, self).__init__(**kwargs)

    def build(self, input_shape):

        self.sigma = self.add_weight(
            name="sigma",
            shape=(),
            initializer=tf.constant_initializer(self._sigma),
            trainable=False,
        )

        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[1], self.output_size),
            initializer=keras.initializers.GlorotNormal(),
            trainable=True,
        )

        super(STM, self).build(input_shape)

    def call(self, x):
        radials, norms2 = self.get_norms_and_activation(x)
        return radials * norms2

    def spread(self, x, expect=False):
        method = "expectation" if expect is True else "argmin"
        radials, _ = self.get_norms_and_activation(x, method=method)
        radials = radials / (self.sigma * np.sqrt(np.pi * 2))
        return radials

    def get_norms_and_activation(self, x, method="argmin"):
        # compute norms
        norms = tf.norm(tf.expand_dims(x, 2) - tf.expand_dims(self.kernel, 0), axis=1)
        norms2 = tf.pow(norms, 2)

        # compute activation
        radials = self.get_radials(norms2, method)
        return radials, norms2

    def backward(self, radials):
        x = tf.matmul(radials, tf.transpose(self.kernel))
        return x

    def get_wta(self, norms2, method="argmin"):

        if method == "expectation":

            nnorms2 = tf.exp(-0.5 * (s2 ** -1) * tf.pow(norms2, 2))
            nnorms2 *= 1 / (0.2 * np.sqrt(2 * np.pi))
            nnorms2 = tf.expand_dims(nnorms2, axis=-1)
            gridmul = tf.multiply(nnorms2, self.grid)
            wta = tf.reduce_mean(gridmul, axis=1)
        elif method == "argmin":
            wta = tf.cast(tf.argmin(norms2, axis=1), dtype="float")
            wta = tf.transpose(tf.stack([wta // self.side, wta % self.side]))

        return wta

    def get_radials(self, norms2, method="argmin"):

        wta = self.get_wta(norms2, method)
        radials = self.radial_fun(wta, self.sigma, self.output_size)
        radials = radials / (self.sigma * np.sqrt(np.pi * 2))

        return radials

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

    def loss(self, radial_norms2, extrinsic=None):
        if extrinsic is None:
            extrinsic = tf.ones_like(radial_norms2)
        return tf.reduce_mean(radial_norms2 * extrinsic, axis=1)


if __name__ == "__main__":

    inp_num = 2
    out_num = 100
    initial_sigma = out_num / 2
    min_sigma = 1
    initial_lr = 1.0
    stime = 10000
    decay_window = stime / 4

    loss = []

    som_layer = STM(out_num, initial_sigma, radial_fun=radial2d, name="SOM")

    # Setting the model

    inputs = keras.layers.Input(shape=[inp_num])
    outputs = som_layer(inputs)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.add_loss(som_layer.loss(outputs))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=initial_lr), metrics=None
    )

    for t in range(stime):
        # learning rate and sigma annealing
        curr_sigma = min_sigma + initial_sigma * np.exp(-t / decay_window)
        curr_rl = initial_lr * np.exp(-t / decay_window)

        # update learning rate and sigma in the graph
        keras.backend.set_value(model.get_layer("SOM").sigma, curr_sigma)
        keras.backend.set_value(model.optimizer.lr, curr_rl)

        data = np.random.uniform(0, 1, [100, 2])
        loss_ = model.train_on_batch(data)
        if t % (stime // 10) == 0:
            print(
                loss_,
            )
        loss.append(loss_)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weights = model.get_layer("SOM").kernel.numpy()
    plt.scatter(*weights)
    plt.savefig("weights.png")
