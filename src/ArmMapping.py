import numpy as np
from stm import STM, radial2d as radial
import tensorflow.keras as keras

import gym, box2dsim

def getData(trials, stime, env):
    data = np.zeros([trials, stime, 2])
    for k in range(trials):
        env.reset()
        action = np.zeros(5)
        action[:3] = np.random.uniform(-1,1,3)
        action[:3] = 0.5*(action[:3]*np.pi*np.array([1.8, 1, 1]) - np.pi)
        print("pos epoch:",k)
        for t in range(stime):
            d = env.step(action)
            data[k, t] = d
    data = data.reshape(trials*stime, -1)
    data = data[(data[:,0]>10) & (data[:,1]>10)]
    np.save("data/StoredArmGenerateData", data)
    return data

def generatePrototypes(data, out_num):
    # parameters
    items = data.shape[0]
    batch_size = 50
    batch_num = items // batch_size
    idcs = np.arange(items)
    inp_num = 2
    initial_sigma = out_num/2
    min_sigma = 0.7
    initial_lr = 2.0
    epochs = 100
    decay_window = epochs/10
    loss = []
    # Setting the model
    som_layer = STM(out_num, initial_sigma, radial_fun=radial, name="SOM")
    inputs = keras.layers.Input(shape=[inp_num])
    outputs = som_layer(inputs)
    model = keras.models.Model(
        inputs=inputs, outputs=outputs)
    model.add_loss(som_layer.loss(outputs))
    model.compile(optimizer=keras.optimizers.Adam(
        learning_rate=initial_lr), metrics=None)
    #training
    for epoch in range(epochs):
        # learning rate and sigma annealing
        curr_sigma = min_sigma + initial_sigma*np.exp(-epoch/decay_window)
        curr_rl = initial_lr*np.exp(-epoch/decay_window)
        # update learning rate and sigma in the graph
        keras.backend.set_value(model.get_layer("SOM").sigma, curr_sigma)
        keras.backend.set_value(model.optimizer.lr, curr_rl)
        # iterate batches
        np.random.shuffle(idcs)
        curr_loss = []
        for batch in range(batch_num):
            batch_range = idcs[np.arange(batch_size*batch,
                batch_size*(1 + batch))]
            curr_data = data[batch_range]
            loss_ = model.train_on_batch(curr_data)
            curr_loss.append(loss_)
        loss.append(np.mean(curr_loss))
        print(epoch, loss[-1])
    weights = model.get_layer("SOM").kernel.numpy()
    return weights

def generateArmMapping(inner_domain_shape, env, trials=1000, stime=50):
    """ Generate a topological mapping"""
    # build dataset
    try:
        data = np.load("data/StoredArmGenerateData.npy")
    except IOError:
        data = getData(trials, stime, env)
    # train SOM
    weights = generatePrototypes(data, inner_domain_shape)
    return weights

if __name__ == "__main__":
    
    data = np.load("data/StoredArmGenerateData.npy")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.subplot(111, aspect="equal")
    plt.xlim([5,25])
    plt.ylim([5,25])
    plt.scatter(*data.T, s=3)
    plt.show()
