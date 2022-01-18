import numpy as np
from stm import STM, radial as radial
import tensorflow.keras as keras
import params
import os

import gym, box2dsim


class Env:

    def __init__(self, box2d_env, **kargs):
        self.b2d_env = box2d_env
        self.b2d_env.set_taskspace(**params.task_space)
        self.render = None
        self.reset()

    def step(self, action):
        observation,*_ = self.b2d_env.step(action)
        if self.render is not None:
            self.b2d_env.render(self.render)
        return observation

    def reset(self, world=None):
        if world is None:
            world = np.random.randint(4)
        observation = self.b2d_env.reset(world)
        if self.render is not None:
            self.b2d_env.render_init(self.render)
        return observation


class OnlineEnv:

    def __init__(self, box2d_env, **kargs):
        plt.ion()
        self.b2d_env = box2d_env
        self.b2d_env.set_taskspace(**params.task_space)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect="equal")
        self.im = self.ax.imshow(np.zeros([10,10]))
        self.reset()

    def step(self, action):
        observation,*_ = self.b2d_env.step(action)
        self.b2d_env.render("human")
        self.im.set_array(self.b2d_env.bground_img)
        return observation

    def reset(self, world=None):
        if world is None:
            world = np.random.randint(4)
        self.b2d_env.set_world(world)
        observation = self.b2d_env.reset()
        self.b2d_env.render("human")
        self.im.set_array(self.b2d_env.bground_img)
        self.fig.canvas.draw()
        return observation

from ArmActuator import Agent as ArmAgent
def getData(trials, stime, env):
    
    agent = ArmAgent(env=None, num_inputs=2, num_hidden=100, num_outputs=3, 
            actuator_map_name="data/StoredArmActuatorMap", 
            actuator_weights_name="data/StoredArmActuatorWeights")
    
    data = np.zeros([trials, stime, 3*10*10 + 4 + 7])
    for k in range(trials):
        env.reset()
        arm_action = agent.step(env.b2d_env.handPosInSpace())
        grip_action = np.random.uniform([0, np.pi/2]) 
        action = np.hstack([arm_action, grip_action])
        print("pos epoch:",k)
        for t in range(stime):

            d = env.step(action)
            data[k, t] = np.hstack([
                d["VISUAL_SENSORS"].ravel(), 
                d["TOUCH_SENSORS"], 
                d["JOINT_POSITIONS"]])
    data = data.reshape(trials*stime, -1)
    np.save("data/StoredGripGenerateData", data)
    return data

class PrototypeGenerator:

    def __init__(self, inp_num, out_num, data, batch_size=50, 
            min_sigma=0.7, initial_lr=2.0, epochs=100):
        self.data = data
        self.items = data.shape[0]
        self.batch_size = batch_size
        self.batch_num = self.items // batch_size
        self.idcs = np.arange(self.items)
        self.out_num = out_num
        self.inp_num = inp_num
        self.initial_sigma = out_num/2
        self.min_sigma = min_sigma
        self.initial_lr = initial_lr
        self.epochs = epochs
        self.decay_window = epochs/10

    def __call__(self):
        # parameters
        data = self.data
        items = self.items
        batch_size = self.batch_size
        batch_num = self.batch_num
        idcs = self.idcs
        out_num = self.out_num
        inp_num = self.inp_num
        initial_sigma = self.initial_sigma
        min_sigma = self.min_sigma
        initial_lr = self.initial_lr
        epochs = self.epochs
        decay_window = self.decay_window

        # Setting the model
        som_layer = STM(out_num, initial_sigma,  name="SOM")
        inputs = keras.layers.Input(shape=[inp_num])
        outputs = som_layer(inputs)
        model = keras.models.Model(
            inputs=inputs, outputs=outputs)
        model.add_loss(som_layer.loss(outputs))
        model.compile(optimizer=keras.optimizers.Adam(
            learning_rate=initial_lr), metrics=None)
        model.summary()
         
        #training
        loss = []
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


def generateGripMapping(inner_domain_shape, env, trials=1000, stime=50):
    """ Generate a topological mapping"""
    # build dataset
    try:
        data = np.load("data/StoredGripGenerateData.npy")
        print("data acquired")
    except IOError:
        data = getData(trials, stime, env)
    
    # train touch SOM and get weights
    visual_inp_shape = 300
    touch_inp_shape = 4
    start = visual_inp_shape
    touch = data[:,start:(start + touch_inp_shape)]
    touch = touch[touch.sum(1)>0,:]
    touchWeights = PrototypeGenerator(touch_inp_shape, inner_domain_shape, touch)()

    # train posture SOM and get weights
    posture_inp_shape = 2
    start = visual_inp_shape + touch_inp_shape + 5
    posture = data[:,start:(start + posture_inp_shape)]
    postureWeights = PrototypeGenerator(posture_inp_shape, inner_domain_shape, posture)()

    # get pos weights from arm learning
    try:
        posWeights = np.load("data/StoredArmActuatorMap.npy")
    except IOError:
        print("warning: ArmActuator map weights not found.")
        posWeights = np.zeros([2, inner_domain_shape])
    
    weights = np.vstack([touchWeights, postureWeights, posWeights])/3

    return weights

if __name__ == "__main__":
    b2d_env = gym.make('Box2DSimOneArmOneEye-v0')
    env = Env(b2d_env)
    num_hidden = 100
    m = generateGripMapping(num_hidden, env)
    np.save("/tmp/StoredGripActuatorMap", m)
