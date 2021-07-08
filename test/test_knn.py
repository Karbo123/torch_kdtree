import torch
import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
import torch_kdtree
from torch_cluster import knn
from time import time


if __name__ == "__main__":
    NUM = int(2**16)
    KNN = 5
    print(f"(python) num = {NUM}, knn = {KNN}")

    ########################################
    data = torch.randn([NUM, 3], device="cuda") * 1000
    t0 = time()
    tree = torch_kdtree.torchBuildCUDAKDTree(data)
    tree.cpu()
    print(f"(python) time for building kdtree, moving to cpu, and verification = {time() - t0}")
    data = data.cpu()

    ########################################
    query = torch.randn(NUM, 3) * 1000
    t0 = time()
    index = tree.search_knn(query, KNN)
    print(f"(python) time for querying on cpu using multithreads = {time() - t0}")

    ########################################
    data_cuda = data.cuda()
    data_query = query.cuda()
    t0 = time()
    index_gt = knn(data_cuda, data_query, k=KNN)
    print(f"(python) time for querying on gpu using torch_cluster = {time() - t0}")

    t0 = time()
    index_gt = knn(data, query, k=KNN)
    print(f"(python) time for querying on cpu using torch_cluster = {time() - t0}")

    ########################################
    assert (torch.diff(index_gt[0]) >= 0).all()
    index_gt = index_gt[1].reshape([-1, KNN])
    wrong_loc = torch.where(index != index_gt)[0]
    print(f"(python) there are {len(wrong_loc)} mismatches in total")

