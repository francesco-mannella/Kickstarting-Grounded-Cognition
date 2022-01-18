import glob
import os, glob
import params
from shutil import copyfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
import pandas as pd
import seaborn as sns
import SMGraphs
import regex

c = [0, 0.5, 1]
colors = np.vstack([x.ravel() for x in np.meshgrid(c, c, c)]).T
colors = colors[:-1]
c = np.reshape(colors[1:], (5, 5, 3))
c = np.transpose(c, (1, 0, 2))
c = np.reshape(c, (25, 3))
colors[1:, :] = c
full_palette = LinearSegmentedColormap.from_list("basic", colors)

internal_side = int(np.sqrt(params.internal_size))
visual_side = int(np.sqrt(params.visual_size / 3))


def generic_map(wfile, ax):
    side = internal_side
    size = params.internal_size
    max_explained_var_ratio = 0.9
    pca = PCA(whiten=True, n_components=3)
    data = np.load(wfile)

    data = data - np.mean(data, 0) / np.std(data, 0)

    pcanalysis = pca.fit(data.T)
    comps = pcanalysis.transform(data.T)
    comps = (comps - comps.min()) / (comps.max() - comps.min())
    ax.imshow(comps.reshape(side, side, -1), aspect="auto")
    ax.set_axis_off()
    return comps


def map_groups(wfile, ax):
    side = internal_side
    size = params.internal_size
    max_explained_var_ratio = 0.9
    pca = PCA(whiten=True, n_components=3)
    data = np.load(wfile)

    data = data - np.mean(data, 0) / np.std(data, 0)
    pcanalysis = pca.fit(data.T)
    comps = pcanalysis.transform(data.T)
    comps = (comps - comps.min()) / (comps.max() - comps.min())
    clustering = KMeans(n_clusters=3).fit(comps)
    vals = clustering.labels_
    vals = vals.reshape(10, 10)
    ax.imshow(vals, cmap=full_palette, aspect="auto")
    ax.set_axis_off()
    return comps, vals


def visual_map(wfile, ax):

    # visual map
    data_v = np.load(wfile)
    data_v = data_v.reshape(visual_side, visual_side, 3, internal_side, internal_side)
    data_v = data_v.transpose(3, 0, 4, 1, 2)
    data_v = data_v.reshape(visual_side * internal_side, visual_side * internal_side, 3)

    ax.imshow((data_v - data_v.min()) / (data_v.max() - data_v.min()))
    ax.set_axis_off()


def ssensory_map(wfile, ax):

    # visual map
    data = np.load(wfile)
    data = (data - data.min()) / (data.max() - data.min())
    data = data.reshape(4, internal_side, internal_side)
    data = data.transpose(0, 2, 1)
    data = data.reshape(4, -1)

    ss = np.array(
        [
            [[0.0, 0.0], [-0.5, 0.25]],
            [[0.0, 0.0], [0.5, 0.25]],
            [[-0.5, 0.25], [-0.2, 0.5]],
            [[0.5, 0.25], [0.2, 0.5]],
        ]
    )

    idx = 0
    for x in range(internal_side):
        for y in range(internal_side):
            for s in range(4):
                ax.plot(
                    *(ss[s] * [0.6, 0.8] + [y, x]).T, c="black", lw=1 + 3 * data[s, idx]
                )
            idx += 1

    ax.set_xlim([-1, internal_side])
    ax.set_ylim([-1, internal_side])
    ax.set_axis_off()


def proprio_map(wfile, ax):

    # visual map
    data = np.load(wfile)
    data = data.reshape(5, internal_side, internal_side)
    data = data.transpose(0, 2, 1)
    data = data.reshape(5, -1)[3:, :].T

    data = data * 0.8

    idx = 0
    for x in range(internal_side):
        for y in range(internal_side - 1, -1, -1):
            a1, a2 = data[idx]
            ss = np.array(
                [
                    [[0.0, 0.0], [np.cos(a1), np.sin(a1)]],
                    [[0.0, 0.0], [np.cos(-a1), np.sin(-a1)]],
                    [
                        [np.cos(a1), np.sin(a1)],
                        [(np.cos(a1) + np.cos(a2)), (np.sin(a1) + np.sin(a2))],
                    ],
                    [
                        [np.cos(-a1), np.sin(-a1)],
                        [(np.cos(-a1) + np.cos(-a2)), (np.sin(-a1) + np.sin(-a2))],
                    ],
                ]
            )

            for s in range(4):
                ax.plot(*(ss * [0.3, 0.3] + [x, y]).T, c="black")
            idx += 1

    ax.set_xlim([-1, internal_side])
    ax.set_ylim([-1, internal_side])
    ax.set_axis_off()


def topological_variance(wfile, ax, k=3):
    side = internal_side
    size = params.internal_size
    max_explained_var_ratio = 0.9
    pca = PCA(whiten=True, n_components=3)
    data = np.load(wfile)

    data = data - np.mean(data, 0) / np.std(data, 0)
    pcanalysis = pca.fit(data.T)
    comps = pcanalysis.transform(data.T)
    comps = (comps - comps.min()) / (comps.max() - comps.min())
    clustering = KMeans(n_clusters=k).fit(comps)
    vals = clustering.labels_
    vals = vals.reshape(10, 10)
    ax.imshow(vals, cmap=ListedColormap(["#880000", "#88EE55", "#CC22EE"]))
    ax.set_axis_off()


def figure_topological_alignement(files=None, n_items=6, every=2):
    if files is None:
        files = [
            sorted(glob.glob(f"storage/{l}*weights*npy"))
            for l in ["visual", "ssensory", "proprio", "policy"]
        ]

    epochs = [int(regex.sub(".*_(\d+).npy", "\\1", f)) for f in sorted(files[0])]
    data = []
    fig = plt.figure(figsize=(n_items, 6))
    gs = gridspec.GridSpec(13, n_items)

    for i, f in enumerate(files):
        row = []
        for k in range(n_items * every):
            j = k // every
            ax = fig.add_subplot(gs[(i * 3) : ((i * 3) + 2), j], aspect="auto")
            ax.set_axis_off()
            d = generic_map(f[k], ax)
            row.append(d)
        data.append(row)
    data = np.array(data).reshape(4, n_items, -1)

    changes = np.linalg.norm(np.diff(data, axis=1), axis=2)
    for i, d in enumerate(data):
        ax = fig.add_subplot(gs[i * 3 + 2, :], aspect="auto")
        plt.plot(np.arange(n_items), np.hstack([np.nan, changes[i]]), c="black")
        plt.scatter(np.arange(n_items), np.hstack([99, changes[i]]), c="black")
        ax.set_xlim([0, n_items])
        ax.set_ylim([changes[i].min() - changes[i].max() * 0.5, changes[i].max() * 1.5])
        ax.set_xticks([])
        ax.set_yticks([0])

    ax = fig.add_subplot(gs[i * 3 + 3, :], aspect="auto")
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_yticks([])
    ticks = epochs[0 : (n_items * every) : every]
    print(ticks)
    ax.set_xlim([0, n_items])
    labels = [str(e) for e in ticks]
    labels[-1] = ""
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(labels=labels)

    fig.tight_layout(pad=0.1)
    fig.savefig("Fig_alignement.svg")


def figure_variance(wfile):
    fig = plt.figure(figsize=(8, 2))
    gs = gridspec.GridSpec(1, 4)
    ax1 = fig.add_subplot(gs[:, 0])
    _ = generic_map(wfile, ax1)
    ax2 = fig.add_subplot(gs[:, 1])
    comps, vals = map_groups(wfile, ax2)

    df = np.hstack([comps, vals.reshape(-1, 1)])
    df = pd.DataFrame(df, columns=["p1", "p2", "p3", "cluster"])

    fig.add_subplot(gs[2:])
    ms = pd.melt(df, id_vars=["cluster"])
    sns.boxplot(x="cluster", y="value", hue="variable", data=ms)
    fig2 = plt.figure()
    fig2.add_subplot(111)
    ms = df.groupby(["cluster"], as_index=False).agg(["mean"]).reset_index()
    mse = df.groupby(["cluster"], as_index=False).agg(["std"]).reset_index()
    ms.plot(x="cluster", y=["p1", "p2", "p3"], yerr=mse, kind="bar", capsize=4, rot=0)


import os


def figure_paths(wfile="epoch_000010.npy"):

    stime = params.stime + 1
    episodes = params.batch_size
    contexts = 4
    idxs = np.arange(stime * episodes * contexts).reshape(-1, 1)

    data_dict = np.load(wfile, allow_pickle=True)[0]
    data = [data_dict[context] for context in ["v_r", "ss_r", "p_r", "a_r"]]

    data = np.vstack(data)
    iepochs = idxs * 0 + 1
    icontexts = idxs // (stime * episodes)
    iepisodes = (idxs // stime) % episodes
    ts = idxs % stime

    data_s = data_dict["ss"].mean(1)
    data_s = data_s.reshape(-1, 1)
    data_s = np.tile(data_s, [4, 1])

    df = pd.DataFrame(
        np.hstack([iepochs, icontexts, iepisodes, ts, data_s.reshape(-1, 1), data]),
        columns=["epoch", "context", "episode", "ts", "ss"]
        + [f"p{x:04d}" for x in range(100)],
    )

    df.head()

    df = df.loc[
        df.groupby("episode", as_index=False)["ss"].filter(lambda x: x.sum() > 0).index
    ]
    ep_idcs = np.unique(df["episode"].to_numpy())

    df["context"] = pd.Categorical(df["context"])

    df["context"] = df["context"].cat.rename_categories(
        {0: "visual", 1: "ssensory", 2: "proprio", 3: "policy"}
    )

    def make_df_components(x):
        data = x.iloc[:, 5:].to_numpy()

        pcs = np.argmax(data, 1)
        pcs = np.vstack([pcs // 10, pcs % 10])

        x["pc1"] = pcs[0]
        x["pc2"] = pcs[1]
        return x

    pcsdf = df.groupby(["context"]).apply(make_df_components)

    p = sns.color_palette("hls", 4)

    figures = []
    for episode in ep_idcs:
        fig = plt.figure(figsize=(10, 7))
        ax = plt.subplot(111, aspect="equal")

        ps = []
        for context in ["visual", "ssensory", "proprio", "policy"]:
            cdf = pcsdf.query(f' context=="{context}" & episode=={episode}')
            # (p,) = plt.plot(cdf["pc1"], cdf["pc2"], lw=3)
            p = plt.scatter(cdf["pc1"], cdf["pc2"], s=(100 + 10000 * cdf["ss"]))
            ps.append(p)

        plt.legend(ps, ["visual", "ssensory", "proprio", "policy"])

        ax.set_xlim([-1, 14])
        ax.set_ylim([-1, 10])
        _ = ax.set_xticks(np.arange(10))
        _ = ax.set_yticks(np.arange(10))

        figures.append(fig)
    return figures


# %%
fig = plt.figure(figsize=(16, 8))
wfile = "storage/visual_weights_0420.npy"

ax1 = fig.add_subplot(121, aspect="equal")
visual_map(wfile, ax1)
ax2 = fig.add_subplot(122, aspect="equal")
topological_variance(wfile, ax2)
fig.savefig("visual_variance.svg")


# %%

fig = plt.figure(figsize=(16, 8))
wfile = "storage/ssensory_weights_0420.npy"
ax1 = fig.add_subplot(121, aspect="equal")
ssensory_map(wfile, ax1)
ax2 = fig.add_subplot(122, aspect="equal")
topological_variance(wfile, ax2)
fig.savefig("ssensory_variance.svg")


# %%

fig = plt.figure(figsize=(16, 8))
wfile = "storage/proprio_weights_0420.npy"
ax1 = fig.add_subplot(121, aspect="equal")
proprio_map(wfile, ax1)
ax2 = fig.add_subplot(122, aspect="equal")
topological_variance(wfile, ax2)
fig.savefig("proprio_variance.svg")


# %%

fig = plt.figure(figsize=(16, 8))
wfile = "storage/policy_weights_0420.npy"
ax1 = fig.add_subplot(121, aspect="equal")
_ = generic_map(wfile, ax1)
ax2 = fig.add_subplot(122, aspect="equal")
topological_variance(wfile, ax2, k=5)
fig.savefig("policy_variance.svg")

# %%
figure_topological_alignement(n_items=8, every=3)
