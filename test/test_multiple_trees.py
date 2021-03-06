import torch
try:
    import torch_kdtree # if built with setuptools
except:
    import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), "../build")) # if built with cmake
    import torch_kdtree

if __name__ == "__main__":
    NUM = int(2**18)
    RADIUS = 0.1
    print(f"(python) num = {NUM}, radius = {RADIUS}")

    ########################################
    tree1 = torch_kdtree.torchBuildCUDAKDTree(torch.randn([NUM, 3], device="cuda:0"))
    tree1.cpu()
    index1, batch1 = tree1.search_radius(torch.randn(NUM, 3), RADIUS)
    print("finished 1")

    ########################################
    tree2 = torch_kdtree.torchBuildCUDAKDTree(torch.randn([NUM, 3], device="cuda:1"))
    tree2.cpu()
    index2, batch2 = tree2.search_radius(torch.randn(NUM, 3), RADIUS)
    print("finished 2")

