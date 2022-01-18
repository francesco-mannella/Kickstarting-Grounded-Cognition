import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import params

def out_fun(x):
    return 1/(1 + tf.exp(-(x - 0.5)*params.predict_ampl))

class SMPredict:

    def __init__(self, inp_num, out_num, lr=0.1, name="predict"): 
        self.inp_num = inp_num
        self.out_num = out_num
        self.lr = lr
        self.name = name
        initializer = tf.keras.initializers.GlorotUniform()

        # Setting the model
        self.inp_layer = keras.layers.Input(shape=(self.inp_num),name=f"{self.name}_input")
        self.out_layer = keras.layers.Dense(self.out_num, 
                name=f"{self.name}_layer",activation=out_fun, 
            kernel_initializer=keras.initializers.zeros(),
            bias_initializer=keras.initializers.zeros()) 

        self.model = keras.Model(inputs=self.inp_layer, outputs=self.out_layer(self.inp_layer), name=self.name)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                loss="mse", metrics=None)
        self.model.build([None, self.inp_num])
        print(self.model.summary())

        self.t = 0

    def update(self, patterns, labels):
        patterns = tf.cast(patterns, dtype="float32")
        self.loss = self.model.train_on_batch(patterns, labels)
        return self.loss

    def weights(self):
        return self.out_layer.get_weights()[0]

    def get_weights(self, weights):
        keras.backend.set_value(self.model.get_layer(f"{self.name}_layer",).kernel, weights)

    def spread(self, inp):
        assert len(inp.shape) == 2
        inp = tf.cast(inp, dtype="float32")
        return self.model.predict(inp)
        #return np.tanh(params.predict_ampl*self.model.predict(inp))

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    inp_num = 2
    out_num = 1
    patterns_size = 10000
    epochs = 150
    
    labels = np.zeros(patterns_size)
    labels[patterns_size//2:] = 1
    patterns = np.vstack([labels, 1-labels]).T \
            + 0.01*np.random.randn(patterns_size, 2)
    predict = SMPredict(inp_num, out_num)
   
    idcs = np.arange(patterns_size)
    for t in range(epochs):
        np.random.shuffle(idcs)
        preds = predict.spread(patterns)
        loss = predict.update(patterns[idcs], labels[idcs])
        
        comp = 2*np.abs(preds - 0.5)
        print(loss, comp.mean())
