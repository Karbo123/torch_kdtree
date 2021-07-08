import torch
import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
import torch_kdtree
from torch_cluster import radius
from scipy.spatial import cKDTree
from time import time
import numpy as np

if __name__ == "__main__":
    NUM = int(2**16)
    RADIUS = 100
    print(f"(python) num = {NUM}, radius = {RADIUS}")

    ########################################
    data = torch.randn([NUM, 3], device="cuda") * 1000
    t0 = time()
    tree = torch_kdtree.torchBuildCUDAKDTree(data)
    tree.cpu().verify()
    print(f"(python) time for building kdtree, moving to cpu, and verification = {time() - t0}")
    data = data.cpu()

    ########################################
    query = torch.randn(NUM, 3) * 1000
    t0 = time()
    index, batch = tree.search_radius(query, RADIUS)
    print(f"(python) time for querying on cpu using multithreads = {time() - t0}")

    ########################################
    data_cuda = data.cuda()
    data_query = query.cuda()
    t0 = time()
    index_gt = radius(data_cuda, data_query, r=RADIUS)
    print(f"(python) time for querying on gpu using torch_cluster = {time() - t0}")

    t0 = time()
    index_gt = radius(data, query, r=RADIUS)
    print(f"(python) time for querying on cpu using torch_cluster = {time() - t0}")

    ########################################
    t0 = time()
    index_gt = cKDTree(data.numpy()).query_ball_point(query.numpy(), r=RADIUS, workers=8)
    print(f"(python) time for querying on cpu using cKDTree with 8 threads = {time() - t0}")

    ########################################
    index_gt = torch.from_numpy(np.concatenate(index_gt)).long()
    wrong_loc = torch.where(index != index_gt)[0]
    print(f"(python) there are {len(wrong_loc)} mismatches in total")

