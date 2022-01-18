import numpy as np
task_space = {"xlim": [-10, 50], "ylim": [-10, 50]}
stime = 200

esn_tau = 5.0
esn_alpha = 0.2
esn_epsilon = 1.0e-30

arm_input = 2
arm_hidden = 100
arm_output = 3
grip_input = 8 
grip_hidden = 100
grip_output = 5

internal_size = 100
visual_size = 300
somatosensory_size = 4
proprioception_size = 5
policy_size = 100 * 5
num_objects = 4

match_sigma = 8
base_match_sigma = 0.5
base_match_sigma = base_match_sigma/match_sigma 
internal_sigma = 2
base_internal_sigma = 0.7
base_internal_sigma = base_internal_sigma/internal_sigma
base_explore_sigma = 0.0
explore_sigma = 0.2
base_lr = 0.0
stm_lr = 0.2
policy_base = np.pi*0.25

match_th = 0.0
match_incr_th = 0.0
predict_lr = 0.001
predict_ampl = 3
reach_grip_prop = 0.1

epochs = 300
batch_size = 20
tests = 12
epochs_to_test = 10
