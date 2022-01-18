import numpy as np
import params
from GripActuator import Agent as GripAgent
from ArmActuator import Agent as ArmAgent

class SMAgent:

    def __init__(self, env, *args, **kargs):
        self.env = env
        self.arm_agent = ArmAgent(env=None, 
                num_inputs=params.arm_input, num_hidden=params.arm_hidden,
                num_outputs=params.arm_output,
                actuator_map_name="data/StoredArmActuatorMap",
                actuator_weights_name="data/StoredArmActuatorWeights", 
                *args, **kargs)
        self.grip_agent = GripAgent(env=self.env, 
                num_inputs=params.grip_input, num_hidden=params.grip_hidden,
                num_outputs=params.grip_output,
                actuator_map_name="data/StoredGripActuatorMap",
                actuator_weights_name="data/StoredGripActuatorWeights", 
                *args, **kargs)
        self.grip_param_shape = self.grip_agent.grip.params.shape
        self.params_size = np.prod(self.grip_param_shape) 

    def step(self, state):
        pos = state["EYE_POS"][::-1]
        touch = state["TOUCH_SENSORS"] 
        joints = state["JOINT_POSITIONS"][3:5] 
        arm_state = state["EYE_POS"][::-1]
        grip_state = np.hstack([pos, touch, joints])
        arm_action = self.arm_agent.step(arm_state) 
        grip_action = self.grip_agent.step(grip_state) 
        
        grip_action[3:] += params.policy_base
        grip_action[:3] *= params.reach_grip_prop
        action = np.hstack([
            arm_action + grip_action[:3], 
            grip_action[3:]])
        return action
    def reset(self):
        self.arm_agent.reset()
        self.grip_agent.reset()

    def updatePolicy(self, params):
        self.grip_agent.updatePolicy(params)

if __name__ == "__main__":
    from SMEnv import Env 

    env = Env()
    env.render="human"
    state = env.reset(1)
    agent = SMAgent(env)
    for t in range(100):
        action = agent.step(state)
        state = env.step(action)
    input()
