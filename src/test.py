import torch, torchkdtree



RANGE_MAX = torch.iinfo(torch.int32).max / 10
RANGE_MIN = torch.iinfo(torch.int32).min / 10




def buildKDTree(data):
    range_min = data.min(dim=0, keepdim=True)[0]
    range_max = data.max(dim=0, keepdim=True)[0]
    data_input = (data - range_min) / (range_max - range_min)
    data_input = data_input * (RANGE_MAX - RANGE_MIN) + RANGE_MIN
    data_input = data_input.to(dtype=torch.int32)
    node_root = torchkdtree.torchBuildCUDAKDTree(data_input)
    return node_root





if __name__ == "__main__":
    data = torch.randn(4194304, 3)
    node_root = buildKDTree(data)

    import ipdb; ipdb.set_trace()




print()




