import torch
import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), "../../build"))
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
    query = _query.cuda()
    
    time_start = time()
    tree = torch_kdtree.torchBuildCUDAKDTree(data)
    if device == "cpu": tree.cpu()
    else: raise
    index = tree.search_nearest(query)
    time_elapsed = time() - time_start

    return (time_elapsed, index)


# def time_our_knn(_data, _query, k_list, device="cpu"):
#     assert device in ["cpu", "cuda"]
#     data = _data.cuda()
#     query = _query.cuda()
#     results = list()

#     for k in k_list:
#         time_start = time()
#         tree = torch_kdtree.torchBuildCUDAKDTree(data)
#         if device == "cpu": tree.cpu()
#         else: raise
#         index = tree.search_knn(query, k)
#         time_elapsed = time() - time_start
#         results.append((time_elapsed, index))

#     return results

# def time_our_radius(_data, _query, radius_list, device="cpu"):
#     assert device in ["cpu", "cuda"]
#     data = _data.cuda()
#     query = _query.cuda()
#     results = list()

#     for r in radius_list:
#         time_start = time()
#         tree = torch_kdtree.torchBuildCUDAKDTree(data)
#         if device == "cpu": tree.cpu()
#         else: raise
#         index, batch = tree.search_radius(query, r)
#         time_elapsed = time() - time_start
#         results.append((time_elapsed, (index, batch)))

#     return results

###################################################

def time_ckdtree_nearest(_data, _query, device="cpu", threads=8):
    assert device in ["cpu"]
    data = _data.numpy()
    query = _query.numpy()
    
    time_start = time()
    index = cKDTree(data).query(query, workers=threads)[1]
    time_elapsed = time() - time_start

    return (time_elapsed, torch.from_numpy(index).long())

# def time_ckdtree_knn(_data, _query, k_list, device="cpu", threads=8):
#     assert device in ["cpu"]
#     data = _data.numpy()
#     query = _query.numpy()
#     results = list()

#     for k in k_list:
#         time_start = time()
#         index = cKDTree(data).query(query, k=k, workers=threads)[1]
#         time_elapsed = time() - time_start
#         results.append((time_elapsed, torch.from_numpy(index).long()))

#     return results

# def time_ckdtree_radius(_data, _query, radius_list, device="cpu", threads=8):
#     assert device in ["cpu"]
#     data = _data.numpy()
#     query = _query.numpy()
#     results = list()

#     for r in radius_list:
#         time_start = time()
#         index = cKDTree(data).query_ball_point(query, r=r, workers=threads)
#         time_elapsed = time() - time_start
#         results.append((time_elapsed, torch.from_numpy(index).long()))

#     return results

###################################################

def make_plot(save_path, numeric, legend):
    for x, label in zip(numeric, legend):
        plt.plot(x, label=label)
    plt.savefig(save_path)


###################################################

if __name__ == "__main__":

    pairs = dict(num=[(2**14, 3),
                      (2**16, 3),
                    #   (2**18, 3),
                    #   (2**20, 3),
                     ],
                #  dim=[(2**16, 3),
                #       (2**16, 8),
                #       (2**16, 16),
                #       (2**16, 32),
                #      ],
                )

    save_dir = os.path.join(os.path.dirname(__file__), "../../fig")

    ########################
    # nearest
    our_nearest = list()
    ckdtree_nearest = list()
    for variable in ["num", "dim"]:
        for num, dim in pairs[variable]:
            data, query = get_data(num, dim)
            our_nearest.append(time_our_nearest(data, query))
            ckdtree_nearest.append(time_ckdtree_nearest(data, query))
        assert all([(pack_our[1] == pack_ckdtree[1]).all() for pack_our, pack_ckdtree in zip(our_nearest, ckdtree_nearest)])
        make_plot(os.path.join(save_dir, f"fig_time_nearest_{variable}.png"),
                  [[pack[0] for pack in our_nearest],
                   [pack[0] for pack in ckdtree_nearest]], 
                  ["ours", "ckdtree"])


