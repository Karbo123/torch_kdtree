#ifndef TORCH_KDTREE_DEF_H_
#define TORCH_KDTREE_DEF_H_

class TorchKDTree
{
public:
    refIdx_t root;
    KdNode* kdNodes;
    float* coordinates;
    sint numPoints;
    sint numDimensions;
    bool is_cuda;

    ///////////////////////////////////////////////////////////////////
    TorchKDTree(sint _numPoints, sint _numDimensions);
    TorchKDTree(TorchKDTree&& _tree);
    ~TorchKDTree();

    ///////////////////////////////////////////////////////////////////
    std::string __repr__();

    TorchKDTree& cpu();

    sint verify();

    KdNode get_root();

    KdNode get_node(sint index);

    void _traverse_and_assign();

    ///////////////////////////////////////////////////////////////////
    refIdx_t _search(const float* query, refIdx_t node);

    template<int dim> inline
    float distance(const float* point_a, const float* point_b);

    template<int dim> inline 
    float distance_plane(const float* point, refIdx_t node);

    ///////////////////////////////////////////////////////////////////
    // nearest search
    template<int dim>
    void _search_nearest(const float* point, int64_t* out_);

    torch::Tensor search_nearest(torch::Tensor points);

    ///////////////////////////////////////////////////////////////////
    // knn search
    template<int dim>
    void _search_knn(const float* point, sint k, int64_t* out_);

    torch::Tensor search_knn(torch::Tensor points, sint k);

    ///////////////////////////////////////////////////////////////////
    // radius search
    template<int dim>
    void _search_radius(const float* point, float radius2, std::vector<sint>& out_);

    std::tuple<torch::Tensor, torch::Tensor> search_radius(torch::Tensor points, float radius);

};

/////////////////////////////////////////////////////////////////////////////////////////////////////

TorchKDTree::TorchKDTree(sint _numPoints, sint _numDimensions): 
                            root(-1), 
                            numPoints(_numPoints), numDimensions(_numDimensions), 
                            is_cuda(true)
{
    kdNodes = new KdNode[_numPoints];
    coordinates = new float[_numPoints * _numDimensions];
    if (kdNodes == nullptr || coordinates == nullptr)
    {
        throw runtime_error("error when allocating host memory");
    }
}


TorchKDTree::TorchKDTree(TorchKDTree&& _tree)
{
    root = _tree.root; _tree.root = -1;
    kdNodes = _tree.kdNodes; _tree.kdNodes = nullptr;
    coordinates = _tree.coordinates; _tree.coordinates = nullptr;
    numPoints = _tree.numPoints; _tree.numPoints = 0;
    numDimensions = _tree.numDimensions;
    is_cuda = _tree.is_cuda;
}

TorchKDTree::~TorchKDTree()
{
    delete[] kdNodes;
    delete[] coordinates;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

std::string TorchKDTree::__repr__()
{
    std::stringstream _str_stream;
    _str_stream << "<TorchKDTree of " 
                << numDimensions << "dims and " 
                << numPoints << "points on device " 
                << (is_cuda ? "CUDA" : "CPU") << ">" << std::endl;
    return _str_stream.str();
}


TorchKDTree& TorchKDTree::cpu()
{
    if (is_cuda)
    {
        // read the KdTree back from GPU
        Gpu::getKdTreeResults(kdNodes, coordinates, numPoints, numDimensions);
        // now kdNodes have values
    }

    // on CUDA
    is_cuda = false;

    // process the whole tree // TODO actually we should process the tree on CUDA @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    _traverse_and_assign();

    return *this;
}

sint TorchKDTree::verify()
{
    if (is_cuda) throw runtime_error("CUDA-KDTree cannot be verified from host");
    sint numberOfNodes = kdNodes[root].verifyKdTree(kdNodes, coordinates, numDimensions, 0); // number of nodes on host
    return numberOfNodes;
}

KdNode TorchKDTree::get_root()
{
    if (is_cuda) throw runtime_error("CUDA-KDTree cannot access root node from host");
    return kdNodes[root];
}

KdNode TorchKDTree::get_node(sint index)
{
    if (is_cuda) throw runtime_error("CUDA-KDTree cannot access node from host");
    return kdNodes[index];
}

void TorchKDTree::_traverse_and_assign()
{
    using node_parent_depth = std::tuple<refIdx_t, refIdx_t, sint>; // current_node, parent_node, depth
    
    // DFS
    refIdx_t node, parent; sint depth;
    std::stack<node_parent_depth> buffer;
    buffer.emplace(node_parent_depth(root, -1, 0)); // NOTE: -1 means no parent
    while (!buffer.empty())
    {
        std::tie(node, parent, depth) = buffer.top();
        buffer.pop();

        KdNode& node_current = kdNodes[node];

        // assign
        node_current.split_dim = depth % numDimensions;
        node_current.parent = parent;
        if (parent >= 0) node_current.brother = (kdNodes[parent].ltChild == node) ? (kdNodes[parent].gtChild) : (kdNodes[parent].ltChild);
        else node_current.brother = -1;
        
        // traverse
        if (node_current.gtChild >= 0) buffer.emplace(node_parent_depth(node_current.gtChild, node, depth + 1)); // push right tree
        if (node_current.ltChild >= 0) buffer.emplace(node_parent_depth(node_current.ltChild, node, depth + 1)); // push left tree
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

refIdx_t TorchKDTree::_search(const float* query, refIdx_t node) // searching down the tree until reaching leaf node
{
    float val, val_node;
    bool has_left_node, has_right_node;
    while (true)
    {
        const KdNode& node_current = kdNodes[node];

        has_left_node = (node_current.ltChild >= 0);
        has_right_node = (node_current.gtChild >= 0);
        if (has_left_node || has_right_node)
        {
            if (has_left_node && has_right_node)
            {
                val = query[node_current.split_dim];
                val_node = coordinates[numDimensions * node_current.tuple + node_current.split_dim];
                if (val < val_node) node = node_current.ltChild;
                else node = node_current.gtChild;
            }
            else if (has_left_node) node = node_current.ltChild;
            else node = node_current.gtChild;
        }
        else break;
    }
    return node;
}

template<int dim> inline
float TorchKDTree::distance(const float* point_a, const float* point_b)
{
    float sum = 0;
    for (sint i = 0; i < dim; ++i)
        sum += POW2(point_a[i] - point_b[i]);
    return sum;
}

template<> inline
float TorchKDTree::distance<0>(const float* point_a, const float* point_b)
{
    float sum = 0;
    for (sint i = 0; i < numDimensions; ++i)
        sum += POW2(point_a[i] - point_b[i]);
    return sum;
}

template<int dim> inline
float TorchKDTree::distance_plane(const float* point, refIdx_t node)
{
    const KdNode& node_plane = kdNodes[node];
    return POW2(point[node_plane.split_dim] - coordinates[dim * node_plane.tuple + node_plane.split_dim]);
}

template<> inline
float TorchKDTree::distance_plane<0>(const float* point, refIdx_t node)
{
    const KdNode& node_plane = kdNodes[node];
    return POW2(point[node_plane.split_dim] - coordinates[numDimensions * node_plane.tuple + node_plane.split_dim]);
}


#endif
