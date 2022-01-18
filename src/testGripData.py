import numpy as np
import matplotlib.pyplot as plt
plt.ion()

data = np.load("data/StoredGripGenerateData.npy")
plt.subplot(111, aspect="equal")
im = plt.imshow(np.zeros([10, 10]))  

for item in data:

    
    im.set_array(item[:300].reshape(10,10,3))
    plt.pause(0.1)
