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
    float scale, offset;

    TorchKDTree(sint _numPoints, sint _numDimensions, float _scale, float _offset): root(nullptr), numPoints(_numPoints), numDimensions(_numDimensions), is_cuda(true), 
                                                                                    scale(_scale), offset(_offset)
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
        scale = _tree.scale;
        offset = _tree.offset;
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
    refIdx_t _search(const KdCoord* query, refIdx_t node) // searching down the tree until reaching leaf node
    {
        sint split_dim;
        KdCoord val, val_node;
        while (true)
        {
            bool has_left_node = (kdNodes[node].ltChild >= 0);
            bool has_right_node = (kdNodes[node].gtChild >= 0);
            if (has_left_node || has_right_node)
            {
                if (has_left_node && has_right_node)
                {
                    split_dim = kdNodes[node].split_dim;
                    val = query[split_dim];
                    val_node = coordinates[numDimensions * node + split_dim];
                    if (val < val_node) node = kdNodes[node].ltChild;
                    else node = kdNodes[node].gtChild;
                }
                else if (has_left_node) node = kdNodes[node].ltChild;
                else node = kdNodes[node].gtChild;
            }
            else break;
        }
        return node;
    }

    // squared distance
    inline 
    KdCoord squared_distance(const KdCoord* point_a, const KdCoord* point_b)
    {
        KdCoord sum = 0;
        for (sint i = 0; i < numDimensions; ++i)
        {
            sum += POW2(point_a[i] - point_b[i]);
        }
        return sum;
    }

    // to plane squared distance
    inline
    KdCoord squared_distance_plane(const KdCoord* point, const KdNode& node)
    {
        return POW2(point[node.split_dim] - coordinates[numDimensions * node.tuple + node.split_dim]);
    }

    // search for a single point
    void _search_nearest(const KdCoord* point, int64_t* out)
    {
        using start_end = std::tuple<KdNode, KdNode>;

        KdCoord dist       = std::numeric_limits<KdCoord>::max();
        KdCoord dist_plane = std::numeric_limits<KdCoord>::max();
        KdCoord dist_best  = std::numeric_limits<KdCoord>::max();
        KdNode  node_best  = _search(point, *root);
        
        // BFS
        std::queue<start_end> buffer;
        KdNode node_start, node_end, node_bro;
        buffer.emplace(start_end(*root, node_best));
        while (!buffer.empty())
        {
            std::tie(node_start, node_end) = buffer.front();
            buffer.pop();

            dist = squared_distance(point, coordinates + numDimensions * node_start.tuple);
            if (dist < dist_best)
            {
                dist_best = dist;
                node_best = node_start;
            }

            while (node_end.tuple != node_start.tuple)
            {
                dist = squared_distance(point, coordinates + numDimensions * node_end.tuple);
                if (dist < dist_best)
                {
                    dist_best = dist;
                    node_best = node_end;
                }

                if (node_end.brother >= 0)
                {
                    dist_plane = squared_distance_plane(point, kdNodes[node_end.parent]);
                    if (dist_plane < dist_best)
                    {
                        node_bro = kdNodes[node_end.brother];
                        buffer.emplace(start_end(node_bro, _search(point, node_bro)));
                    }
                }

                node_end = kdNodes[node_end.parent]; // back track
            }
        }

        *out = int64_t(node_best.tuple);
    }

    torch::Tensor search_nearest(torch::Tensor points)
    {
        CHECK_CONTIGUOUS(points);
        CHECK_FLOAT32(points);
        TORCH_CHECK(points.size(1) == numDimensions, "dimensions mismatch");
        sint numQuery = points.size(0);
        torch::Tensor points_int = torch::round(points * scale + offset).to(torch::kInt32);
        KdCoord* points_int_ptr = points_int.data_ptr<KdCoord>();
        torch::Tensor indices_tensor;

        if (is_cuda)
        {
            throw runtime_error("CUDA-KDTree cannot do searching");
        }
        else
        {
            indices_tensor = torch::zeros({numQuery}, torch::kInt64);
            int64_t* raw_ptr = indices_tensor.data_ptr<int64_t>();
            
             // 10x faster
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i)
            {
                _search_nearest(points_int_ptr + i * numDimensions, raw_ptr + i);
            }
        }

        return indices_tensor;
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
TorchKDTree torchBuildCUDAKDTree(torch::Tensor data, float scale, float offset)
{
    // check input
    CHECK_CONTIGUOUS(data);
    CHECK_INT32(data);
    sint numPoints = data.size(0);
    sint numDimensions = data.size(1);
    TorchKDTree tree(numPoints, numDimensions, scale, offset);
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
        .def_readonly("scale", &TorchKDTree::scale)
        .def_readonly("offset", &TorchKDTree::offset)
        .def_property_readonly("root", &TorchKDTree::get_root)
        .def("cpu", &TorchKDTree::cpu)
        .def("verify", &TorchKDTree::verify)
        .def("node", &TorchKDTree::get_node)
        .def("__repr__", &TorchKDTree::__repr__)
        .def("search_nearest", &TorchKDTree::search_nearest);

}



/*
    TODO: KDTree search @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        https://zhuanlan.zhihu.com/p/45346117
        https://bbs.huaweicloud.com/blogs/169897

        https://stackoverflow.com/questions/34688977/how-do-i-traverse-a-kdtree-to-find-k-nearest-neighbors

        https://en.wikipedia.org/wiki/K-d_tree#Nearest_neighbour_search

*/
