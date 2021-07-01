#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "torch_kdtree.h"

#include <string>
#include <sstream>
#include <ostream>
#include <stdexcept>

namespace py = pybind11;

////////////////////////////////////////////////////////////////

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.dtype()==torch::kFloat32, #x " must be float32")

//////////

namespace CUDA_ERR
{
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
};
#define gpuErrchk(ans)                                  \
    {                                                   \
        CUDA_ERR::gpuAssert((ans), __FILE__, __LINE__); \
    }

////////////////////////////////////////////////////////////////

std::string environ_cuda = "";
const sint numGPUs       = 1;
const sint numThreads    = 512;
const sint numBlocks     = 32;

/*
    build CUDA-KDTree
    TODO 
    - if it is already on CUDA, do not copy to CPU
*/
KdNode torchBuildCUDAKDTree(torch::Tensor data, bool print_message)
{
    // check input
    CHECK_CONTIGUOUS(data);
    CHECK_FLOAT32(data);
    sint numPoints = data.size(0);
    sint numDimensions = data.size(1);
    bool is_cuda = data.is_cuda();
    KdCoord* coordinates = nullptr;
    if (is_cuda)
    {
        KdCoord* data_ptr = data.data_ptr<KdCoord>();
        coordinates = new KdCoord[numPoints * numDimensions]; // on HOST
        gpuErrchk(cudaMemcpy(coordinates, data_ptr, numPoints * numDimensions * sizeof(KdCoord), cudaMemcpyDeviceToHost));
    } else
    {
        coordinates = data.data_ptr<KdCoord>(); // NOTE: do not make a copy on CPU
    }

    // initialize environment
    std::stringstream _str_stream;
    _str_stream << "numGPUs="       << numGPUs       << ", ";
    _str_stream << "numThreads="    << numThreads    << ", ";
    _str_stream << "numBlocks="     << numBlocks     << ", ";
    _str_stream << "numDimensions=" << numDimensions;
    std::string environ_cuda_target = _str_stream.str();
    if (environ_cuda != environ_cuda_target)
    {
        Gpu::gpuSetup(numGPUs, numThreads, numBlocks, numDimensions);
        if (Gpu::getNumThreads() == 0 || Gpu::getNumBlocks() == 0) 
        {
            _str_stream.str("");
            _str_stream << "KdNode Tree cannot be built with " << numThreads << " threads or " << numBlocks << " blocks." << std::endl;
            throw runtime_error(_str_stream.str());
        }
        if (print_message)
        {
            std::cout << "numPoints=" << numPoints << ", "
                      << "numDimensions=" << numDimensions << ", "
                      << "numThreads=" << numThreads << ", "
                      << "numBlocks=" << numBlocks << std::endl;
        }
        environ_cuda = environ_cuda_target;
    }

    // Create the k-d tree.  First copy the data to a tuple in its kdNode.
	// also null out the gt and lt references
	// create and initialize the kdNodes array
	KdNode *kdNodes = new KdNode[numPoints];
    if (kdNodes == NULL) 
    {
        _str_stream.str("");
		_str_stream << "Can't allocate " << numPoints << " kdNodes" << std::endl;
        throw runtime_error(_str_stream.str());
    }

    // create the tree
    KdNode *root = KdNode::createKdTree(kdNodes, coordinates, numDimensions, numPoints);
    
    return (*root);
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

    py::class_<KdNode>(m, "KdNode");
}
