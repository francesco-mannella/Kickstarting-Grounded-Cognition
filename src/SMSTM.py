import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from stm import STM, radial2d


class SMSTM:
    def __init__(self, inp_num, out_num, lr=2.0, sigma=None, min_sigma=0.7, name="stm"):
        self.inp_num = inp_num
        self.out_num = out_num
        self.min_sigma = min_sigma
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = out_num / 2
        self.lr = lr
        self.name = name
        initializer = tf.keras.initializers.GlorotUniform()
        self.stm_layer = STM(
            self.out_num, self.sigma, radial_fun=radial2d, name=self.name
        )
        self.out_side = int(np.sqrt(out_num))
        x = np.arange(self.out_side)
        self.radial_grid = np.stack(np.meshgrid(x, x)).reshape(2, -1).T

        # Setting the model

        self.dists = keras.layers.Input(shape=[self.out_num], name=f"dists_{self.name}")
        self.inputs = keras.layers.Input(shape=[self.inp_num], name=f"input_{self.name}")
        self.outputs = self.stm_layer(self.inputs)
        self.model = keras.models.Model(
            inputs=[self.inputs, self.dists], 
            outputs=self.outputs,
            name=f"model_{self.name}",
        )
        self.model.add_loss(self.stm_layer.loss(self.outputs, self.dists))
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr), metrics=None
        )
        self.model.build([None, [self.inp_num, self.out_num]])
        print(self.model.summary())

        self.t = 0

    def updateParams(self, sigma=None, lr=None):
        # update learning rate and sigma in the graph
        if sigma is not None:
            keras.backend.set_value(self.model.get_layer(self.name).sigma, self.sigma)
        if lr is not None:
            keras.backend.set_value(self.model.optimizer.lr, self.lr)
    
    def update(self, data, dists, sigma=None, lr=None):


        assert len(data.shape) == 2
        assert data.shape[1] == self.inp_num
        data = tf.cast(data, dtype=tf.float32)
        dists = tf.cast(dists, dtype=tf.float32)
        self.loss = self.model.train_on_batch([data, dists])
        return self.loss

    def weights(self):
        return self.model.get_layer(self.name).kernel.numpy()

    def get_weights(self, weights):
        keras.backend.set_value(self.model.get_layer(self.name).kernel, weights)

    def spread(self, inp):
        inp = tf.cast(inp, tf.float32)
        out = self.stm_layer.spread(inp)
        return out

    def getPoint(self, out):
        idx = np.argmax(out, 1)
        point = 1.0*np.vstack([idx // self.out_side, idx % self.out_side]).T 
        return point

    def getRepresentation(self, point, sigma=None):
        return self.interpolate(point, sigma)

    def interpolate(self, point, sigma=None):
        distance = point.reshape(-1, 1, 2) - self.radial_grid.reshape(1, -1, 2)
        distance = np.linalg.norm(distance, axis=2)
        if sigma is None: sigma = self.getSigma() 
        rep = np.exp(-0.5 * (sigma ** -2) * (distance) ** 2)
        rep *= 1/((sigma**2)*np.pi*2)
        return rep

    def backward(self, out):
        out = tf.cast(out, dtype=tf.float32)
        return self.stm_layer.backward(out)

    def getSigma(self):
        return self.model.get_layer(self.name).sigma


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    inp_num = 2
    out_num = 100
    stime = 10000
    decay_window = stime / 8

    stm = SMSTM(inp_num, out_num)
    initial_sigma = stm.sigma
    initial_lr = stm.lr
    min_sigma = stm.min_sigma

    plt.ion()
    scatter = plt.scatter(0, 0)
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1])

    for t in range(stime):
        # learning rate and sigma annealing
        curr_sigma = min_sigma + initial_sigma * np.exp(-t / decay_window)
        curr_rl = initial_lr * np.exp(-t / decay_window)

        data = 0.3 * np.random.randn(10000, 2)
        loss = stm.update(data, curr_sigma, curr_rl)

        if t % 100 == 0:
            w = stm.weights()
            scatter.set_offsets(w.T)
            plt.pause(0.01)

        print(loss)
