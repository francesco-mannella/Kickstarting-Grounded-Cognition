import numpy as np
import matplotlib.pyplot as plt
import os, glob
import mkvideo


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, aspect="auto")

# %% visual weights
vm = mkvideo.vidManager(fig)
vw_files = glob.glob("storage/*visual*")
for f in sorted(vw_files):
    ax.clear()  
    d = np.load(f)
    d = d.reshape(10, 10, 3, 10, 10).transpose(3, 0, 4, 1, 2)
    d = d.reshape(100, 100, 3)
    ax.imshow(d)
    ax.set_axis_off()
    plt.pause(0.1)
    vm.save_frame()
vm.mk_video("visual_weights", ".")

# %% ss weights
vm = mkvideo.vidManager(fig)
vw_files = glob.glob("storage/*ssensory*")
for f in sorted(vw_files):
    d = np.load(f)
    d = d.reshape(40, 10).T
    ax.imshow(d, aspect="auto")
    ax.set_axis_off()
    plt.pause(0.1)
    vm.save_frame()
vm.mk_video("ssensory_weights", ".")
