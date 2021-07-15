# A CUDA implementation of KDTree in PyTorch

> Adapted from: [KdTreeGPU](https://github.com/johnarobinson77/KdTreeGPU)

This repo is specially useful if the point cloud is very large (>100,000 points).

Currently KD-Tree is built on CUDA, and the query is done on CPU.
We are now working on making a new function of querying point on CUDA device, which should be faster. 

Functions currently implemented:
- nearest search (CPU)
- knn search (CPU)
- radius search (CPU)


**NOTE: this repo is still under heavy development**


# Build

build environment: (other environment should be okey)
- torch == 1.8.0
- nvcc == 10.2

```
mkdir build && cd build

cmake .. \
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
-DCMAKE_CUDA_ARCHITECTURES=60 \
-DCUDA_TOOLKIT_ROOT_DIR=$CU102_CUDA_TOOLKIT_DIR
```

# usage

please check the testing script in `test/performance/` folder.


# benchmarking

**nearest search**
![](fig/fig_time_nearest.png)


# TODO

- [x] multiple trees memory conflict
- [x] remove all global variables such as `d_verifyKdTreeError`
- [ ] CUDA query
- [ ] support any num of points
- [ ] host memory leak testing

other ref: [Traversal on CUDA](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/)

