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
#define CHECK_INT32(x) TORCH_CHECK(x.dtype()==torch::kInt32, #x " must be int32")

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

class TorchKDTree
{
public:
    KdNode* root;
    KdNode* kdNodes;
    KdCoord* coordinates;
    sint numPoints;
    sint numDimensions;
    bool is_cuda;

    TorchKDTree(sint _numPoints, sint _numDimensions): root(nullptr), numPoints(_numPoints), numDimensions(_numDimensions), is_cuda(true)
    {
        kdNodes = new KdNode[numPoints];
        coordinates = new KdCoord[numPoints * numDimensions];
        if (kdNodes == nullptr || coordinates == nullptr)
        {
            throw runtime_error("error when allocating host memory");
        }
    }

    TorchKDTree(TorchKDTree&& _tree)
    {
        root = _tree.root; _tree.root = nullptr;
        kdNodes = _tree.kdNodes; _tree.kdNodes = nullptr;
        coordinates = _tree.coordinates; _tree.coordinates = nullptr;
        numPoints = _tree.numPoints; _tree.numPoints = 0;
        numDimensions = _tree.numDimensions;
        is_cuda = _tree.is_cuda;
    }

    ~TorchKDTree()
    {
        delete[] kdNodes;
        delete[] coordinates;
    }

    std::string __repr__()
    {
        std::stringstream _str_stream;
        _str_stream << "<TorchKDTree of " << numDimensions << "dims and " << numPoints << "points on device " << (is_cuda ? "CUDA" : "CPU") << ">" << std::endl;
        return _str_stream.str();
    }

    TorchKDTree& cpu()
    {
        // read the KdTree back from GPU
        Gpu::getKdTreeResults(kdNodes, coordinates, numPoints, numDimensions);
        // now kdNodes have values
        is_cuda = false;
        return *this;
    }

    sint verify()
    {
        if (is_cuda) throw runtime_error("CUDA-KDTree cannot be verified from host");
        sint numberOfNodes = root->verifyKdTree(kdNodes, coordinates, numDimensions, 0); // number of nodes on host
        return numberOfNodes;
    }

    KdNode get_root()
    {
        if (is_cuda) throw runtime_error("CUDA-KDTree cannot access root node from host");
        return *root;
    }

    sint get_root_index()
    {
        if (is_cuda) throw runtime_error("CUDA-KDTree cannot access root index from host");
        return root - kdNodes;
    }

    KdNode get_node(sint index)
    {
        if (is_cuda) throw runtime_error("CUDA-KDTree cannot access node from host");
        return kdNodes[index];
    }
};

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
TorchKDTree torchBuildCUDAKDTree(torch::Tensor data)
{
    // check input
    CHECK_CONTIGUOUS(data);
    CHECK_INT32(data);
    sint numPoints = data.size(0);
    sint numDimensions = data.size(1);
    TorchKDTree tree(numPoints, numDimensions);
    if (data.is_cuda())
    {
        KdCoord* data_ptr = data.data_ptr<KdCoord>();
        gpuErrchk(cudaMemcpy(tree.coordinates, data_ptr, numPoints * numDimensions * sizeof(KdCoord), cudaMemcpyDeviceToHost)); // make a copy
    } else
    {
        KdCoord* data_ptr = data.data_ptr<KdCoord>();
        std::memcpy(tree.coordinates, data_ptr, numPoints * numDimensions * sizeof(KdCoord)); // make a copy
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
            _str_stream << "KdNode Tree cannot be built with " << numThreads << " threads or " << numBlocks << " blocks" << std::endl;
            throw runtime_error(_str_stream.str());
        }
        
        std::cout << "numPoints=" << numPoints << ", "
                  << "numDimensions=" << numDimensions << ", "
                  << "numThreads=" << numThreads << ", "
                  << "numBlocks=" << numBlocks << ", "
                  << "numGPUs=" << numGPUs << std::endl;
        
        environ_cuda = environ_cuda_target;
    }

    // create the tree
    // NOTE:
    // - kdNodes unchanges
    tree.root = KdNode::createKdTree(tree.kdNodes, tree.coordinates, numDimensions, numPoints);
    
    return std::move(tree); // no copy
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "A CUDA implementation of KDTree in PyTorch";

    m.def("torchBuildCUDAKDTree", &torchBuildCUDAKDTree);

    py::class_<KdNode>(m, "KdNode")
        .def(py::init<int32_t, refIdx_t, refIdx_t>()) // tuple, ltChild, gtChild
        .def_readwrite("tuple", &KdNode::tuple)
        .def_readwrite("ltChild", &KdNode::ltChild)
        .def_readwrite("gtChild", &KdNode::gtChild);
    
    py::class_<TorchKDTree>(m, "TorchKDTree")
        .def_readonly("numPoints", &TorchKDTree::numPoints)
        .def_readonly("numDimensions", &TorchKDTree::numDimensions)
        .def_readonly("is_cuda", &TorchKDTree::is_cuda)
        .def("cpu", &TorchKDTree::cpu)
        .def("verify", &TorchKDTree::verify)
        .def("get_root", &TorchKDTree::get_root)
        .def("get_node", &TorchKDTree::get_node)
        .def("__repr__", &TorchKDTree::__repr__);

}
