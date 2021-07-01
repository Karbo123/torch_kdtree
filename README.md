# A CUDA implementation of KDTree in PyTorch

> Adapted from: [KdTreeGPU](https://github.com/johnarobinson77/KdTreeGPU)

under heavy development

# Build

build environment:
- torch == 1.8.0
- nvcc == 10.2

```
mkdir build && cd build

cmake .. \
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
-DCMAKE_CUDA_ARCHITECTURES=60 \
-DCUDA_TOOLKIT_ROOT_DIR=$CU102_CUDA_TOOLKIT_DIR
```
