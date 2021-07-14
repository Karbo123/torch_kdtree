
#ifndef TORCH_KDTREE_KNN_H_
#define TORCH_KDTREE_KNN_H_

class cmp_dist_node_less
{
    using dist_node = std::tuple<float, refIdx_t>;

public:
    bool operator() (const dist_node& a, const dist_node& b)
    {
        return std::get<0>(a) < std::get<0>(b);
    };
};


// search for a single point
template<int dim>
void TorchKDTree::_search_knn(const float* point, sint k, int64_t* out_)
{
    using start_end = std::tuple<refIdx_t, refIdx_t>;
    using dist_node = std::tuple<float, refIdx_t>;

    float dist       = std::numeric_limits<float>::max();
    float dist_plane = std::numeric_limits<float>::max();
    refIdx_t node_bro;

    // create a queue
    const auto _init_cont = std::vector<dist_node>(k, dist_node(std::numeric_limits<float>::max(), -1));
    auto heap_best = std::priority_queue<dist_node, std::vector<dist_node>, cmp_dist_node_less> (_init_cont.begin(), _init_cont.end());
    
    // BFS
    std::queue<start_end> buffer;
    refIdx_t node_start, node_end;
    buffer.emplace(start_end(root, _search(point, root)));

    while (!buffer.empty())
    {
        std::tie(node_start, node_end) = buffer.front();
        buffer.pop();

        // back trace until to the starting node
        while (true)
        {
            // update if current node is better
            dist = distance<dim>(point, coordinates + numDimensions * kdNodes[node_end].tuple);
            if (dist < std::get<0>(heap_best.top())) // exists a smaller value
            {
                heap_best.pop(); // remove the largest elem from heap
                heap_best.emplace(dist_node(dist, node_end)); // record the best
            }

            if (node_end != node_start)
            {
                node_bro = kdNodes[node_end].brother;
                if (node_bro >= 0)
                {
                    // if intersect with plane, search another branch
                    dist_plane = distance_plane<dim>(point, kdNodes[node_end].parent);
                    if (dist_plane < std::get<0>(heap_best.top()))
                    {
                        buffer.emplace(start_end(node_bro, _search(point, node_bro)));
                    }
                }

                // back trace
                node_end = kdNodes[node_end].parent;
            }
            else break;
        }
    }

    // copy results
    for (sint i = 0; i < k; ++i)
    {
        out_[k - 1 - i] = int64_t(kdNodes[std::get<1>(heap_best.top())].tuple); // out_[0] is the nearest, out_[k - 1] is the farthest
        heap_best.pop();
    }
}



template<int N>
struct WorkerKnn
{
    static void work(TorchKDTree* tree_ptr, float* points_ptr, int64_t* raw_ptr, int numDimensions, int numQuery, int k)
    {
        #pragma omp parallel for
        for (sint i = 0; i < numQuery; ++i) tree_ptr->_search_knn<N>(points_ptr + i * numDimensions, k, raw_ptr + i * k);
    }
};

torch::Tensor TorchKDTree::search_knn(torch::Tensor points, sint k)
{
    CHECK_CONTIGUOUS(points);
    CHECK_FLOAT32(points);
    TORCH_CHECK(points.size(1) == numDimensions, "dimensions mismatch");
    sint numQuery = points.size(0);

    float* points_ptr = points.data_ptr<float>();
    torch::Tensor indices_tensor;

    if (is_cuda)
    {
        throw runtime_error("CUDA-KDTree cannot do searching");
    }
    else
    {
        indices_tensor = torch::zeros({numQuery, k}, torch::kInt64);
        int64_t* raw_ptr = indices_tensor.data_ptr<int64_t>();
        
        Dispatcher<WorkerKnn>::dispatch(numDimensions,
                                        this, points_ptr, raw_ptr, numDimensions, numQuery, k);
    }

    return indices_tensor;
}

#endif