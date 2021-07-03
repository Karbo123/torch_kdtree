import torch
import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
import torchkdtree
from scipy.spatial.ckdtree import cKDTreeNode


# NOTE 3 * ((MAX - MIN) / K) ** 2 <= MAX     ==>   K >= 160529.75978303837
RANGE_MAX = torch.iinfo(torch.int32).max / 160530
RANGE_MIN = torch.iinfo(torch.int32).min / 160530


def buildKDTree(data):
    range_min, range_max = data.min(), data.max()
    scale = (RANGE_MAX - RANGE_MIN) / (range_max - range_min)
    offset = -range_min * scale + RANGE_MIN
    data_input = torch.round(data * scale + offset).to(dtype=torch.int32)
    tree = torchkdtree.torchBuildCUDAKDTree(data_input.clone())
    return tree, data_input


# def TorchKDTree_to_cKDTree(tree, data):

#     dim = data.size(1)
    
#     def _convert(_node, level, split_dim):
#         split = data[_node.tuple, split_dim]
#         lesser = _node.ltChild
#         greater = _node.gtChild

#         return cKDTreeNode(level=level, 
#                            split=split,
#                            lesser=_convert(tree.get_node(lesser), level + 1, (split_dim + 1) % dim) if lesser >=0 else None,
#                            greater=_convert(tree.get_node(greater), level + 1, (split_dim + 1) % dim) if greater >=0 else None,
#                         )

#     cKDTreeRoot = _convert(tree.get_root(), 0, 0)

#     import ipdb; ipdb.set_trace()




if __name__ == "__main__":
    # data = torch.randn(4194304, 3)
    data = torch.randn(65536, 3)
    tree, data_input = buildKDTree(data)
    tree.cpu().verify()
    import ipdb; ipdb.set_trace()

    for i in range(65536):
        assert i == tree.test_search_nearest(data_input[i])
        print(f"okey for {i}")


    # TorchKDTree_to_cKDTree(tree, data)

    # import ipdb; ipdb.set_trace()


print()




""" [benchmarking]
import torch
from time import time
from torch_cluster import knn
x = torch.randn([262144, 3], device="cuda")
y = torch.randn([262144, 3], device="cuda")
t0 = time(); edges = knn(x, y, k=1); print(time() - t0)            # takes 34.24 sec. on a single 2080Ti


import torch
from time import time
from torch_cluster import knn
x = torch.randn([262144, 3], device="cpu")
y = torch.randn([262144, 3], device="cpu")
t0 = time(); edges = knn(x, y, k=1); print(time() - t0)            # takes 0.78 sec. on a free CPU


"""