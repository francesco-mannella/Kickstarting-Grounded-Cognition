import numpy as np
import params
from esn import ESN
import gym, box2dsim

from GripMapping import generateGripMapping, Env as GripEnv
generateMapping = generateGripMapping

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

class GripActuator:
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
                print("{:} not found".format(actuator_weights_name))

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
        self.env = env
        self.grip = GripActuator(env, *args, **kargs)
        self.num_params = self.grip.num_hidden*self.grip.num_outputs

    def step(self, state):
        out = self.grip.step(state)
        out = 0.5*np.pi*(out)

        return out

    def reset(self):
        self.grip.reset()

    def updatePolicy(self, params):
        self.grip.params = np.reshape(params,
            [self.grip.num_hidden, self.grip.num_outputs])

if __name__ == "__main__":
    import gym, box2dsim
    from ArmActuator import Agent as ArmAgent, Env as ArmEnv
    env = gym.make('Box2DSimOneArmOneEye-v0')

    arm_num_inputs = 2
    arm_num_hidden = 100
    arm_num_outputs = 3
    arm_agent = ArmAgent(ArmEnv(env), num_inputs=arm_num_inputs,
            num_hidden=arm_num_hidden, num_outputs=arm_num_outputs,
            actuator_map_name="data/StoredArmActuatorMap",
            actuator_weights_name="data/StoredArmActuatorWeights")

    grip_num_inputs = 4 + 2 + 2
    grip_num_hidden = 100
    grip_num_outputs = 5
    grip_agent = Agent(GripEnv(env), num_inputs=grip_num_inputs,
            num_hidden=grip_num_hidden, num_outputs=grip_num_outputs,
            actuator_map_name="data/StoredGripActuatorMap",
            actuator_weights_name="data/StoredGripActuatorWeights")
    grip_agent.updatePolicy(np.pi*(.1*np.ones([grip_num_hidden, grip_num_outputs])))

    for t in range(100000):
        if t%100 == 0:
            o = grip_agent.env.reset()
            arm_action = arm_agent.step([15,15]) + 0.5*np.random.randn(arm_num_outputs)

        grip_action = grip_agent.step(
                np.hstack([
                    [0, 0],
                    o["TOUCH_SENSORS"], 
                    o["JOINT_POSITIONS"][3:5]]))
        action = np.hstack([
            arm_action + grip_action[:3]*0.4, grip_action[3:]])
        o = grip_agent.env.step(action)
        env.render("human")
    input()
