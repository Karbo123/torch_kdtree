
#include "torch_kdtree_utils.h"
#include "torch_kdtree_func.h"
#include "torch_kdtree_def.h"
#include "torch_kdtree_nearest.h"
#include "torch_kdtree_knn.h"


/*
    build CUDA-KDTree
    TODO  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    - if it is already on CUDA, do not copy to CPU
*/
TorchKDTree torchBuildCUDAKDTree(torch::Tensor data_float)
{
    // check input
    CHECK_CONTIGUOUS(data_float);
    CHECK_FLOAT32(data_float);
    sint numPoints = data_float.size(0);
    sint numDimensions = data_float.size(1);

    auto val_max = std::get<0>(torch::max(data_float, 0, false));
    auto val_min = std::get<0>(torch::min(data_float, 0, false));
    const float bound_scaling = 0.1; // NOTE: too large this value may cause error in `cudaMemcpyFromSymbolAsync` ???
    float int_max = std::numeric_limits<KdCoord>::max() * bound_scaling;
    float int_min = std::numeric_limits<KdCoord>::min() * bound_scaling;
    auto scale  = float(int_max - int_min) / (val_max - val_min);
    auto offset = -val_min * scale + int_min;
    auto data = torch::round(data_float * scale + offset).to(torch::kInt32);

    TorchKDTree tree(numPoints, numDimensions);
    if (data.is_cuda())
    {
        KdCoord* data_ptr = data.data_ptr<KdCoord>();
        gpuErrchk(cudaMemcpy(tree.coordinates, data_ptr, numPoints * numDimensions * sizeof(KdCoord), cudaMemcpyDeviceToHost)); // make a copy
        float* float_ptr = data_float.data_ptr<float>();
        gpuErrchk(cudaMemcpy(tree.coordinates_float, float_ptr, numPoints * numDimensions * sizeof(float), cudaMemcpyDeviceToHost)); // make a copy
    } else
    {
        KdCoord* data_ptr = data.data_ptr<KdCoord>();
        std::memcpy(tree.coordinates, data_ptr, numPoints * numDimensions * sizeof(KdCoord)); // make a copy
        float* float_ptr = data_float.data_ptr<float>();
        std::memcpy(tree.coordinates_float, float_ptr, numPoints * numDimensions * sizeof(float)); // make a copy
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
    KdNode* root_ptr = KdNode::createKdTree(tree.kdNodes, tree.coordinates, numDimensions, numPoints);
    tree.root = refIdx_t(root_ptr - tree.kdNodes);

    return std::move(tree); // no copy
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "A CUDA implementation of KDTree in PyTorch";

    m.def("torchBuildCUDAKDTree", &torchBuildCUDAKDTree);

    py::class_<KdNode>(m, "KdNode")
        .def(py::init<int32_t, refIdx_t, refIdx_t>()) // tuple, ltChild, gtChild
        .def_readonly("tuple", &KdNode::tuple)
        .def_readonly("ltChild", &KdNode::ltChild)
        .def_readonly("gtChild", &KdNode::gtChild)
        // 
        .def_readonly("split_dim", &KdNode::split_dim)
        .def_readonly("parent", &KdNode::parent)
        .def_readonly("brother", &KdNode::brother);
    
    py::class_<TorchKDTree>(m, "TorchKDTree")
        // attribute
        .def_readonly("numPoints", &TorchKDTree::numPoints)
        .def_readonly("numDimensions", &TorchKDTree::numDimensions)
        .def_readonly("is_cuda", &TorchKDTree::is_cuda)
        .def("__repr__", &TorchKDTree::__repr__)

        // access node
        .def_property_readonly("root", &TorchKDTree::get_root)
        .def("node", &TorchKDTree::get_node)

        // cpu
        .def("cpu", &TorchKDTree::cpu)
        .def("verify", &TorchKDTree::verify)
        .def("search_nearest", &TorchKDTree::search_nearest)
        .def("search_knn", &TorchKDTree::search_knn);

}

