import numpy as np
import params
from esn import ESN
import gym, box2dsim

from ArmMapping import generateArmMapping
generateMapping = generateArmMapping

def grid(side):
    x = np.arange(side)
    Z = np.stack(np.meshgrid(x, x)).reshape(2, -1).T
    return Z

class Radial:

    def __init__(self, m):
        inp, out = m.shape
        self.l = int(np.sqrt(out))
        self.s = self.l/5
        self.Z = grid(self.l)

    def gauss(self, m, s):
        d = np.linalg.norm(self.Z - m, axis=1)
        return np.exp(-0.5*(s**-2)*d**2)

    def normalize(self, x):
        return x/((self.s**2)*2*np.pi)

    def normal(self, x):
        return self.normalize(self.gauss(x, self.s))

    def __call__(self, x, s=None):

        self.s = s if s is not None else self.s
        mx = np.argmax(x)
        m =  [mx//self.l, mx%self.l]
        return self.normal(m)

class ArmActuator:
    def __init__(self, env, actuator_map_name=None, 
            actuator_weights_name=None, **kargs):
        self.num_inputs = kargs["num_inputs"]
        self.num_hidden = kargs["num_hidden"]
        self.num_outputs = kargs["num_outputs"]
        self.params = np.zeros([self.num_hidden, self.num_outputs])
        self.use_esn = False
        self.grid = None
        self.side_hidden = int(np.sqrt(self.num_hidden))

        if actuator_map_name is not None:
            try:
                self.map =  np.load(actuator_map_name+".npy")
            except IOError:
                self.map = generateMapping(self.num_hidden, env)
                np.save(actuator_map_name, self.map)
                print("Map Saved")
        if actuator_weights_name is not None:
            try:
                self.params =  np.load(actuator_weights_name+".npy")
                self.params = self.params.reshape(self.num_hidden, self.num_outputs)
            except IOError:
                print("Warning: {:} not found".format(actuator_weights_name))

        self.radial = Radial(self.map)

        if "rng" in kargs:
            self.rng = kargs["rng"]
        else:
            self.rng = np.random.RandomState()

        self.hidden_func = lambda x: x
        if self.use_esn:
            self.echo = ESN(N=num_hidden,
                    stime=params.stime,
                    dt=1.0,
                    tau=params.esn_tau,
                    alpha=params.esn_alpha,
                    epsilon=params.esn_epsilon,
                    rng=self.rng)
            self.hidden_func = self.echo.step

    def step(self, state):
        mapped_inp = np.dot(state, self.map) 
        hidden = self.hidden_func(self.radial(mapped_inp))
        out = np.dot(hidden, self.params)
        out = np.maximum(-1, np.minimum(1, out))
        return out

    def reset(self):
        if self.use_esn:
            self.echo.reset()

    def interpolate(self, pos):
        if self.grid is None:
            t = np.arange(self.side_hidden)
            self.grid = np.stack(np.meshgrid(t, t)).reshape(2,-1).T
        ppos = self.grid[int(pos*(self.num_hidden-1))] 

        smooth = self.radial.normal(ppos)
        smooth = smooth/smooth.sum()
        inp = np.dot(self.map, smooth)
        return inp

class Agent:

    def __init__(self, env, *args, **kargs):
        self.arm = ArmActuator(env, *args, **kargs)
        self.num_params = self.arm.num_hidden*self.arm.num_outputs

    def step(self, state):
        out = self.arm.step(state)
        out[:3] = 0.5*(out[:3]*np.pi*np.array([1.8, 1, 1]) - np.pi)

        return out

    def reset(self):
        self.arm.reset()

    def updatePolicy(self, params):
        self.arm.params = np.reshape(params,
            [self.arm.num_hidden, self.arm.num_outputs])

class Env:
    def __init__(self, box2d_env, **kargs):
        self.b2d_env = box2d_env
        self.b2d_env.set_taskspace(**params.task_space)
        self.render = None
        self.rng = self.b2d_env.rng
        self.reset()

    def step(self, action):
        self.b2d_env.step(action)
        if self.render is not None:
            self.b2d_env.render(self.render)
        return self.b2d_env.handPosInSpace()

    def reset(self):
        self.b2d_env.reset(self.b2d_env.worlds["noobject"])
        if self.render is not None:
            self.b2d_env.render(self.render)
        return self.b2d_env.handPosInSpace()

if __name__ == "__main__":
    import gym, box2dsim
    env = gym.make('Box2DSimOneArmOneEye-v0')
    env_ = Env(env)

    agent = Agent(env=None, num_inputs=2, num_hidden=100, num_outputs=3, 
            actuator_map_name="data/StoredArmActuatorMap", 
            actuator_weights_name="data/StoredArmActuatorWeights")
    #agent.updatePolicy(1.2*np.ones([100, 3]))
    env.render_init("human")
    for t in range(100000):
        if t%200 == 0: 
            env.reset()
            action = agent.step([15,15]) + 0.5*np.random.randn(3)
        env.step(np.hstack([action, [0,0]]))
        env.render("human")
    input()
