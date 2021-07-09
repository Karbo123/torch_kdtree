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

setting: build with 262144 points, query with 262144 points.

**nearest search**
```
(python) num = 262144
numPoints=262144, numDimensions=3, numThreads=512, numBlocks=32
0 equal nodes removed. 
Number of nodes = 262144
totalTime = 0.0071  initTime = 0.0014  sortTime + removeDuplicatesTime = 0.0026  kdTime = 0.0017  verifyTime = 0.0014
(python) time for building kdtree, moving to cpu, and verification = 0.03239560127258301
(python) time for querying on cpu using multithreads = 0.05979561805725098
(python) time for querying on gpu using torch_cluster = 34.12433195114136
(python) time for querying on cpu using torch_cluster = 0.6985323429107666
(python) there are 0 mismatches in total
```

**knn search**
```
(python) num = 262144, knn = 5
numPoints=262144, numDimensions=3, numThreads=512, numBlocks=32
0 equal nodes removed. 
Number of nodes = 262144
totalTime = 0.0076  initTime = 0.0013  sortTime + removeDuplicatesTime = 0.0029  kdTime = 0.0017  verifyTime = 0.0017
(python) time for building kdtree, moving to cpu, and verification = 0.033992767333984375
(python) time for querying on cpu using multithreads = 0.14211273193359375
(python) time for querying on gpu using torch_cluster = 61.90901041030884
(python) time for querying on cpu using torch_cluster = 1.3592958450317383
(python) there are 0 mismatches in total
```

**radius search**
```
(python) num = 262144, radius = 100
numPoints=262144, numDimensions=3, numThreads=512, numBlocks=32
0 equal nodes removed. 
Number of nodes = 262144
totalTime = 0.0067  initTime = 0.0010  sortTime + removeDuplicatesTime = 0.0026  kdTime = 0.0017  verifyTime = 0.0014
(python) time for building kdtree, moving to cpu, and verification = 0.03257036209106445
(python) time for querying on cpu using multithreads = 0.19965219497680664
(python) time for querying on gpu using torch_cluster = 28.944889783859253
(python) time for querying on cpu using torch_cluster = 2.196791410446167
(python) time for querying on cpu using cKDTree with 8 threads = 1.28279709815979
(python) there are 0 mismatches in total
```

# TODO

- [ ] multiple trees memory conflict
- [ ] remove all global variables such as `d_verifyKdTreeError`
- [ ] CUDA query

other ref: [Traversal on CUDA](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/)








<!-- ---

global variables to be removed:
```
d_partitionError                                              [okey]

d_verifyKdTreeError                                           [okey]

skc_error
d_RanksA     d_RanksB     d_LimitsA     d_LimitsB
maxSampleCount
d_mpi
d_pivot

d_removeDupsCount
d_removeDupsError
d_removeDupsErrorAdr

gpu
``` -->

