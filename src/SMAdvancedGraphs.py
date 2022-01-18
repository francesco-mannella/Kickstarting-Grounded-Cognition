import glob
import os
import params
from shutil import copyfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mkvideo import vidManager


from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

c = [0, 0.5, 1]
colors = np.vstack([x.ravel() for x in np.meshgrid(c, c, c)]).T
colors = colors[:-1]
c = np.reshape(colors[1:], (5, 5, 3))
c = np.transpose(c, (1, 0, 2))
c = np.reshape(c, (25, 3))
colors[1:,:] = c
full_palette = LinearSegmentedColormap.from_list("basic", colors)

internal_side = int(np.sqrt(params.internal_size))
visual_side = int(np.sqrt(params.visual_size / 3))

def generic_map(wfile):
    side = internal_side
    size = params.internal_size
    max_explained_var_ratio = 0.9
    pca = PCA(whiten=True, n_components=2)
    data = np.load("weights.npy")

    data = data - np.mean(data, 0)/np.std(data, 0)

    pcanalysis = pca.fit(data)
    comps = pcanalysis.transform(data)

    diffs =  comps.reshape(1, -1, size) - comps.reshape(-1, 1, size)
    norms = np.linalg.norm(diffs, axis=-1)
    eps = np.std(norms)/np.sqrt(len(norms.ravel()))
    eps /= 10
    print(f"eps: {eps:8.6f}")
    clustering = DBSCAN(eps=eps, min_samples=10).fit(comps)
    data = clustering.labels_.reshape(side, side)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")
    ax.imshow(data, cmap=full_palette)
    ax.set_axis_off()
    fig.tight_layout(pad=0.0)
    fig.savefig(wfile.split(".")[0]+".png")
    plt.close("all")

