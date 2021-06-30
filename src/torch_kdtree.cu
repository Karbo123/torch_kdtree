#include "torch_kdtree.h"
#include <torch/extension.h>

namespace py = pybind11;



/*
    build CUDA-KDTree (TODO float32)
*/
KdNode torchBuildCUDAKDTree(torch::Tensor data)
{
    return KdNode();
}



/*
    move the CUDA-KDTree to CPU
*/
void torchKDTree2CPU()
{
}



/*
    ball query CPU-KDTree
*/
void torchBallQueryCPUKDTree(float radius, int num_max)
{
}



/*
    knn query CPU-KDTree
*/
void torchKNNQueryCPUKDTree(int k)
{
}


/*
    ball query CUDA-KDTree
*/
void torchBallQueryCUDAKDTree(float radius, int num_max)
{
}


/*
    knn query CUDA-KDTree
*/
void torchKNNQueryCUDAKDTree(int k)
{
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "A CUDA implementation of KDTree in PyTorch";

    m.def("torchBuildCUDAKDTree", &torchBuildCUDAKDTree);
    m.def("torchKDTree2CPU", &torchKDTree2CPU);
    m.def("torchBallQueryCPUKDTree", &torchBallQueryCPUKDTree);
    m.def("torchKNNQueryCPUKDTree", &torchKNNQueryCPUKDTree);
    m.def("torchBallQueryCUDAKDTree", &torchBallQueryCUDAKDTree);
    m.def("torchKNNQueryCUDAKDTree", &torchKNNQueryCUDAKDTree);
}
