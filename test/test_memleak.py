""" testing whether cuda memory leaks
"""
import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
import torch, torch_kdtree
import pynvml # https://pypi.org/project/pynvml/

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

data = torch.randn([2**25, 3], device="cuda")
del data; torch.cuda.empty_cache()
memory_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used

data = [torch.randn([2**20, 3], device="cuda") for _ in range(64)]
tree = [torch_kdtree.torchBuildCUDAKDTree(d) for d in data]
del tree, data; torch.cuda.empty_cache()
memory_after = pynvml.nvmlDeviceGetMemoryInfo(handle).used

print("=======================================")
memory_before /= 2**20
memory_after /= 2**20
memory_error = abs(memory_before - memory_after)
print(f"memory_before = {memory_before:.3f}MB")
print(f"memory_after = {memory_after:.3f}MB")
if memory_error <= 10:
    print(f"cuda memory allocation is fine, no more than {memory_error:.3f} MB")
else:
    print("memory leakage is larger than 10MB")
