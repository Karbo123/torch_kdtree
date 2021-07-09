
#include "torch_kdtree_utils.h"
#include "torch_kdtree_func.h"
#include "torch_kdtree_def.h"
#include "torch_kdtree_nearest.h"
#include "torch_kdtree_knn.h"
#include "torch_kdtree_radius.h"


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
    CHECK_CUDA(data_float);
    sint numPoints = data_float.size(0);
    sint numDimensions = data_float.size(1);
    sint gpuId = data_float.device().index();

    TorchKDTree tree(numPoints, numDimensions);
    
    // copy coordinates to host // TODO do not copy for speed
    float* float_ptr = data_float.data_ptr<float>();
    gpuErrchk(cudaMemcpy(tree.coordinates, float_ptr, numPoints * numDimensions * sizeof(float), cudaMemcpyDeviceToHost));
    
    // initialize environment
    std::stringstream _str_stream;
    _str_stream << "numThreads="    << numThreads    << ", ";
    _str_stream << "numBlocks="     << numBlocks     << ", ";
    _str_stream << "numDimensions=" << numDimensions;
    std::string environ_cuda_target = _str_stream.str();
    if (tree.environ_cuda != environ_cuda_target)
    {
        tree.device = Gpu::gpuSetup(numThreads, numBlocks, gpuId, numDimensions);
        if (tree.device->getNumThreads() == 0 || tree.device->getNumBlocks() == 0) 
        {
            _str_stream.str("");
            _str_stream << "KdNode Tree cannot be built with " << numThreads << " threads or " << numBlocks << " blocks" << std::endl;
            throw runtime_error(_str_stream.str());
        }
        
        std::cout << "numPoints=" << numPoints << ", "
                  << "numDimensions=" << numDimensions << ", "
                  << "numThreads=" << numThreads << ", "
                  << "numBlocks=" << numBlocks << std::endl;
        
        tree.environ_cuda = environ_cuda_target;
    }

    // create the tree
    // NOTE:
    // - kdNodes unchanges
    KdNode* root_ptr = Gpu::createKdTree(tree.device, tree.kdNodes, tree.coordinates, numDimensions, numPoints);
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
        .def("search_nearest", &TorchKDTree::search_nearest)
        .def("search_knn", &TorchKDTree::search_knn)
        .def("search_radius", &TorchKDTree::search_radius);

}

