""" CUDA_VISIBLE_DEVICES=7 python ../test/performance/test_plot.py
"""
import torch
try:
    import torch_kdtree # if built with setuptools
except:
    import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), "../../build")) # if built with cmake
    import torch_kdtree
from torch_cluster import radius
from scipy.spatial import cKDTree
from time import time
import numpy as np
import matplotlib.pyplot as plt

###################################################

def get_data(num, dim):
    data = torch.randn(num, dim)
    query = torch.randn(num, dim)
    return data, query

###################################################

def time_our_nearest(_data, _query, device="cpu"):
    assert device in ["cpu", "cuda"]
    data = _data.cuda()
    query = _query.cuda() if device == "cuda" else _query.clone()
    
    time_start = time()
    tree = torch_kdtree.torchBuildCUDAKDTree(data)
    if device == "cpu": tree.cpu()
    else: raise
    index = tree.search_nearest(query)
    time_elapsed = time() - time_start

    return (time_elapsed, index)


###################################################

def time_ckdtree_nearest(_data, _query, device="cpu", threads=8):
    assert device in ["cpu"]
    data = _data.numpy()
    query = _query.numpy()
    
    time_start = time()
    index = cKDTree(data).query(query, workers=threads)[1]
    time_elapsed = time() - time_start

    return (time_elapsed, torch.from_numpy(index).long())


###################################################

cnt_subplot = 0
def make_plot(save_path, numeric, legend, title, xlabel, xticks):
    global cnt_subplot
    if cnt_subplot % 2 == 0: plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, cnt_subplot % 2 + 1); cnt_subplot += 1
    for x, label in zip(numeric, legend):
        plt.plot(x, label=label)
    plt.legend()
    plt.title(title)
    plt.ylabel("time (sec.)")
    plt.xlabel(xlabel)
    plt.xticks(np.arange(len(xticks)), xticks)
    if save_path: plt.savefig(save_path)


###################################################

if __name__ == "__main__":

    pairs = dict(num=[(2**16, 3),
                      (2**18, 3),
                      (2**20, 3),
                     ],
                 dim=[(2**18, 3),
                      (2**18, 5),
                      (2**18, 8)
                     ],
                )

    save_dir = os.path.join(os.path.dirname(__file__), "../../fig")

    ########################
    # nearest
    for variable in ["num", "dim"]:
        our_nearest = list()
        ckdtree_nearest = list()
        for num, dim in pairs[variable]:
            data, query = get_data(num, dim)
            our_nearest.append(time_our_nearest(data, query)); print("ours is okey")
            ckdtree_nearest.append(time_ckdtree_nearest(data, query)); print("ckdtree is okey")
        assert all([(pack_our[1] == pack_ckdtree[1]).all() for pack_our, pack_ckdtree in zip(our_nearest, ckdtree_nearest)])
        make_plot(os.path.join(save_dir, "fig_time_nearest.png") if variable == "dim" else None,
                  [[pack[0] for pack in our_nearest],
                   [pack[0] for pack in ckdtree_nearest]], 
                  ["ours with cpu query", "ckdtree with 8 threads"],
                  title="nearest search",
                  xlabel="number of queries" if variable == "num" else "dimensions",
                  xticks=[r"$2^{16}$", r"$2^{18}$", r"$2^{20}$"] if variable == "num" else [r"3", r"5", r"8"])


