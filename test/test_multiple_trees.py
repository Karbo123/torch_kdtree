import torch
import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
import torch_kdtree

if __name__ == "__main__":
    NUM = int(2**18)
    RADIUS = 0.1
    print(f"(python) num = {NUM}, radius = {RADIUS}")

    ########################################
    tree = torch_kdtree.torchBuildCUDAKDTree(torch.randn([NUM, 3], device="cuda:0"))
    tree.cpu()
    index, batch = tree.search_radius(torch.randn(NUM, 3), RADIUS)
    print("finished 1")

    ########################################
    tree2 = torch_kdtree.torchBuildCUDAKDTree(torch.randn([NUM, 3], device="cuda:1")) ###### the second card does not run
    tree2.cpu()
    index2, batch2 = tree2.search_radius(torch.randn(NUM, 3), RADIUS)
    print("finished 2")

    import ipdb; ipdb.set_trace()

print()
