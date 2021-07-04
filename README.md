# A CUDA implementation of KDTree in PyTorch

> Adapted from: [KdTreeGPU](https://github.com/johnarobinson77/KdTreeGPU)

This repo is specially useful if the point cloud is very large (>100,000 points).

Currently KD-Tree is built on CUDA, and the query is done on CPU.
We are now working on making a new function of querying point on CUDA device, which should be faster. 

Functions currently implemented:
- nearest search (CPU)


**NOTE** this repo is still under heavy development


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

please check the testing script in `test/test.py`

# benchmarking

setting: build with N points, query with N points.


N = 2**15 = 32768
```
(python) num = 32768
Checking for multiple GPUs...
CUDA-capable device count: 1
> GPU0 = "GeForce RTX 2080 Ti" IS  capable of Peer-to-Peer (P2P)
numPoints=32768, numDimensions=3, numThreads=512, numBlocks=32, numGPUs=1
0 equal nodes removed. 
Number of nodes = 32768
totalTime = 0.0023  initTime = 0.0005  sortTime + removeDuplicatesTime = 0.0008  kdTime = 0.0005  verifyTime = 0.0003

(python) time for building kdtree, moving to cpu, and verification = 0.14187383651733398
(python) time for querying on cpu using multithreads = 0.008480310440063477
(python) time for querying on gpu using torch_cluster = 4.0668065547943115
(python) time for querying on cpu using torch_cluster = 0.05574607849121094
(python) there are 0 mismatches in total
```

N = 2**18 = 262144
```
(python) num = 262144
Checking for multiple GPUs...
CUDA-capable device count: 1
> GPU0 = "GeForce RTX 2080 Ti" IS  capable of Peer-to-Peer (P2P)
numPoints=262144, numDimensions=3, numThreads=512, numBlocks=32, numGPUs=1
0 equal nodes removed. 
Number of nodes = 262144
totalTime = 0.0083  initTime = 0.0016  sortTime + removeDuplicatesTime = 0.0034  kdTime = 0.0017  verifyTime = 0.0016

(python) time for building kdtree, moving to cpu, and verification = 0.19263243675231934  (much faster !!!)
(python) time for querying on cpu using multithreads = 0.05649924278259277                (much faster !!!)
(python) time for querying on gpu using torch_cluster = 37.511107206344604
(python) time for querying on cpu using torch_cluster = 0.6656770706176758
(python) there are 0 mismatches in total
```

N = 2**20 = 1048576
```
(python) num = 1048576
Checking for multiple GPUs...
CUDA-capable device count: 1
> GPU0 = "GeForce RTX 2080 Ti" IS  capable of Peer-to-Peer (P2P)
numPoints=1048576, numDimensions=3, numThreads=512, numBlocks=32, numGPUs=1
0 equal nodes removed. 
Number of nodes = 1048576
totalTime = 0.0284  initTime = 0.0037  sortTime + removeDuplicatesTime = 0.0091  kdTime = 0.0127  verifyTime = 0.0028

(python) time for building kdtree, moving to cpu, and verification = 0.4154829978942871      (much faster !!!)
(python) time for querying on cpu using multithreads = 0.31452035903930664                   (much faster !!!)
(python) time for querying on gpu using torch_cluster = 564.4491164684296
(python) time for querying on cpu using torch_cluster = 4.288048028945923
(python) there are 0 mismatches in total
```

