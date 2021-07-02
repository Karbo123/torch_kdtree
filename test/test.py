import torch
import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
import torchkdtree

RANGE_MAX = torch.iinfo(torch.int32).max / 10
RANGE_MIN = torch.iinfo(torch.int32).min / 10

def buildKDTree(data):
    range_min, range_max = data.min(), data.max()
    scale = (RANGE_MAX - RANGE_MIN) / (range_max - range_min)
    offset = -range_min * scale + RANGE_MIN
    data_input = torch.round(data * scale + offset).to(dtype=torch.int32)
    tree = torchkdtree.torchBuildCUDAKDTree(data_input.clone())
    return tree, data_input


if __name__ == "__main__":
    data = torch.randn(4194304, 3)
    tree, data_input = buildKDTree(data)
    tree.cpu().verify()
    import ipdb; ipdb.set_trace()




print()




