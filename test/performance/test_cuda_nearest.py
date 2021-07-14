import torch
import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), "../../build"))
import torch_kdtree
from torch_cluster import knn
from time import time


if __name__ == "__main__":
    NUM = int(2**18)
    print(f"(python) num = {NUM}")

    ########################################
    data = torch.randn([NUM, 3], device="cuda")
    t0 = time()
    tree = torch_kdtree.torchBuildCUDAKDTree(data)
    print(f"(python) time for building kdtree, and moving to cpu = {time() - t0}")

    ########################################
    query = torch.randn([NUM, 3], device="cuda")
    t0 = time()
    index = tree.search_nearest(query)
    print(f"(python) time for querying on cpu using multithreads = {time() - t0}")
    index = index.cpu()
    
    ########################################

    data = data.cpu()
    query = query.cpu()
    t0 = time()
    index_gt = knn(data, query, k=1)
    print(f"(python) time for querying on cpu using torch_cluster = {time() - t0}")

    ########################################
    index_gt = index_gt[1][torch.argsort(index_gt[0])]
    wrong_loc = torch.where(index != index_gt)[0]
    print(f"(python) there are {len(wrong_loc)} mismatches in total")

