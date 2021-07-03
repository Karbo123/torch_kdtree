#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "torch_kdtree.h"

#include <string>
#include <sstream>
#include <ostream>
#include <stdexcept>
#include <stack>
#include <tuple>

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

#define POW2(x) ((x) * (x))

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
        // process the whole tree // TODO actually we should process the tree on CUDA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        _traverse_and_assign();
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

    void _traverse_and_assign()
    {
        using node_parent_depth = std::tuple<refIdx_t, refIdx_t, sint>; // current_node, parent_node, depth
        
        // DFS
        refIdx_t node, parent; sint depth;
        std::stack<node_parent_depth> buffer;
        buffer.emplace(node_parent_depth(root - kdNodes, -1, 0)); // NOTE: -1 means no parent
        while (!buffer.empty())
        {
            std::tie(node, parent, depth) = buffer.top();
            buffer.pop();

            // do something
            kdNodes[node].split_dim = depth % numDimensions;
            kdNodes[node].parent = parent;
            if (parent >= 0) kdNodes[node].brother = (kdNodes[parent].ltChild == node) ? (kdNodes[parent].gtChild) : (kdNodes[parent].ltChild);
            else kdNodes[node].brother = -1;
            
            
            if (kdNodes[node].gtChild >= 0) buffer.emplace(node_parent_depth(kdNodes[node].gtChild, node, depth + 1)); // push right tree
            if (kdNodes[node].ltChild >= 0) buffer.emplace(node_parent_depth(kdNodes[node].ltChild, node, depth + 1)); // push left tree
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////// searching functions ///////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////
    // search for one query point, from _node of depth to the bottom leaf
    KdNode _search(const KdCoord* query, const KdNode& _node, sint depth)
    {
        KdNode node = _node; // finally to be leaf node
        sint depth_inc = 0;
        while (true)
        {
            bool has_left_node = (node.ltChild >= 0);
            bool has_right_node = (node.gtChild >= 0);
            if (has_left_node || has_right_node)
            {
                if (!has_left_node) node = get_node(node.gtChild);
                else if (!has_right_node) node = get_node(node.ltChild);
                else
                {
                    sint split_dim = (depth + depth_inc) % numDimensions;
                    KdCoord val = query[split_dim];
                    KdCoord val_node = coordinates[numDimensions * node.tuple + split_dim];
                    if (val < val_node) node = get_node(node.ltChild);
                    else node = get_node(node.gtChild);
                }
                depth_inc++;
            }
            else break;
        }
        return node;
    }

    // squared distance
    KdCoord squared_distance(KdCoord* point_a, KdCoord* point_b, sint numDimensions)
    {
        KdCoord sum = 0;
        for (sint i = 0; i < numDimensions; ++i)
        {
            sum += POW2(point_a[i] - point_b[i]);
        }
        return sum;
    }

    // to plane squared distance
    KdCoord squared_distance_plane(KdCoord* point, sint split_dim, const KdNode& _node)
    {
        return POW2(point[split_dim] - coordinates[numDimensions * _node.tuple + split_dim]);
    }


};

////////////////////////////////////////////////////////////////

std::string environ_cuda = "";
const sint numGPUs       = 1;
const sint numThreads    = 512;
const sint numBlocks     = 32;



/*
    build CUDA-KDTree
    TODO  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
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
        .def_readonly("tuple", &KdNode::tuple)
        .def_readonly("ltChild", &KdNode::ltChild)
        .def_readonly("gtChild", &KdNode::gtChild)
        .def_readonly("split_dim", &KdNode::split_dim)
        .def_readonly("parent", &KdNode::parent)
        .def_readonly("brother", &KdNode::brother);
    
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



/*
    TODO: KDTree search @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        https://zhuanlan.zhihu.com/p/45346117
        https://bbs.huaweicloud.com/blogs/169897

        https://stackoverflow.com/questions/34688977/how-do-i-traverse-a-kdtree-to-find-k-nearest-neighbors

        https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search

*/
