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

c = [0, 0.5, 1]
colors = np.vstack([x.ravel() for x in np.meshgrid(c, c, c)]).T
colors = colors[:-1]
c = np.reshape(colors[1:], (5, 5, 3))
c = np.transpose(c, (1, 0, 2))
c = np.reshape(c, (25, 3))
colors[1:, :] = c
full_palette = LinearSegmentedColormap.from_list("basic", colors)

palette = matplotlib.colors.LinearSegmentedColormap.from_list(
    name="custom", colors=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]
)
internal_side = int(np.sqrt(params.internal_size))
visual_side = int(np.sqrt(params.visual_size / 3))


def remove_figs(epoch=0):
    if epoch > 0:
        copyfile("visual_map.png", "storage/visual_map_%08d.png" % epoch)
        copyfile("comp_map.png", "storage/comp_map_%08d.png" % epoch)
        copyfile("log.png", "storage/log_%08d.png" % epoch)
        try:
            copyfile("trajectories.png", "storage/trajectories_%08d.png" % epoch)
        except OSError:
            pass
    else:
        print("Starting simulation ...")
        if not (os.path.exists("blank.gif")):
            blank_video()
        os.makedirs("storage", exist_ok=True)
        for f in glob.glob("storage/*"):
            os.remove(f)
        copyfile("blank.gif", "tv.gif")

    figs = glob.glob("episode*.gif") + glob.glob("*.png")
    for f in figs:
        os.remove(f)
    for k in range(params.tests):
        copyfile("blank.gif", "episode%d.gif" % k)

    copyfile("blank.gif", "visual_map.png")
    copyfile("blank.gif", "comp_map.png")
    copyfile("blank.gif", "log.png")
    copyfile("blank.gif", "trajectories.png")


def trajectories_map(wfile="trajectories.npy"):
    data = np.load(wfile)
    cells, stime, _ = data.shape
    side = int(np.sqrt(cells))
    fig = plt.figure(figsize=(8, 8))
    colors = palette(np.linspace(0, 1, stime))
    for cell in range(cells):
        ax = fig.add_subplot(side, side, cell + 1, aspect="equal")

        ax.add_collection(
            LineCollection(
                segments=np.hstack(
                    [
                        data[cell].reshape(-1, 1, 2)[:-1],
                        data[cell].reshape(-1, 1, 2)[1:],
                    ]
                ),
                colors=colors,
            )
        )
        ax.scatter(*data[cell].T, c=palette(np.linspace(0, 1, stime)), alpha=0.1)

        ax.set_xlim([-0.1, np.pi / 2 + 0.1])
        ax.set_ylim([-0.1, np.pi / 2 + 0.1])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(pad=0.0)
    fig.savefig("trajectories.png")


def visual_map(wfile="visual_weights.npy"):
    # visual map
    data_v = np.load(wfile)
    data_v = data_v.reshape(visual_side, visual_side, 3, internal_side, internal_side)
    data_v = data_v.transpose(3, 0, 4, 1, 2)
    data_v = data_v.reshape(visual_side * internal_side, visual_side * internal_side, 3)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")
    ax.imshow((data_v - data_v.min()) / (data_v.max() - data_v.min()))
    ax.set_axis_off()
    fig.tight_layout(pad=0.0)
    fig.savefig("visual_map.png")
    plt.close("all")


def somatosensory_map(wfile="ssensory_weights.npy"):
    # visual map
    data_v = np.load(wfile)
    data_v = data_v.reshape(4, internal_side, internal_side)
    data_v = data_v.transpose(1, 2, 0)
    data_v = data_v.reshape(internal_side, internal_side * 4)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")
    ax.imshow((data_v - data_v.min()) / (data_v.max() - data_v.min()))
    ax.set_axis_off()
    fig.tight_layout(pad=0.0)
    fig.savefig("ssensory_map.png")
    plt.close("all")


def comp_map(wfile="comp_grid.npy"):
    # comp map
    data_c = np.load("comp_grid.npy")
    data_c = data_c.reshape(internal_side, internal_side)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")
    ax.imshow(data_c, vmin=0, vmax=1)
    ax.set_axis_off()
    fig.tight_layout(pad=0.0)
    fig.savefig("comp_map.png")
    plt.close("all")


def representations_movements(v_r, ss_r, p_r, a_r, name):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")
    vm = vidManager(fig, "irep", "irep", duration=50)

    x = np.arange(internal_side)
    grid = np.stack(np.meshgrid(x, x)).reshape(2, -1)

    pv = ax.scatter(*grid, c="green", s=np.ones(params.internal_size))
    pss = ax.scatter(*grid, c="red", s=100 * np.ones(params.internal_size))
    pp = ax.scatter(*grid, c="blue", s=100 * np.ones(params.internal_size))
    pa = ax.scatter(*grid, c="black", s=100 * np.ones(params.internal_size))

    for i, (v, ss, p, a) in enumerate(zip(v_r, ss_r, p_r, a_r)):
        pv.set_sizes(700 * v)
        pss.set_sizes(700 * ss)
        pp.set_sizes(700 * p)
        pa.set_sizes(700 * a)
        ax.set_title("%d" % i)
        vm.save_frame()

    vm.mk_video(name=name, dirname=".")
    plt.close("all")


def blank_video():
    name = "blank"
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")
    vm = vidManager(fig, "blank_video", "blank_video", duration=50)

    x = np.arange(internal_side)
    grid = np.stack(np.meshgrid(x, x)).reshape(2, -1)

    ax.set_visible(False)

    for t in range(5):
        vm.save_frame()

    vm.mk_video(name=name, dirname=".")
    plt.close("all")


def log(wfile="log.npy"):
    log = np.load(wfile)
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(111)
    stime = len(log)
    ax.fill_between(np.arange(stime), log[:, 0], log[:, 2], fc="red", alpha=0.3)
    ax.plot(np.arange(stime), log[:, 1], c=[0.5, 0, 0])
    ax.set_xlim([-stime * 0.1, stime * 1.1])
    m = log.max()
    ax.set_ylim([-m * 0.1, m * 1.1])
    fig.savefig("log.png")
    plt.close("all")
