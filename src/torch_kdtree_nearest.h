
#ifndef TORCH_KDTREE_NEAREST_H_
#define TORCH_KDTREE_NEAREST_H_

// search for a single point
// see: https://zhuanlan.zhihu.com/p/45346117
template<int dim>
void TorchKDTree::_search_nearest(const float* point, int64_t* out_)
{
    using start_end = std::tuple<refIdx_t, refIdx_t>;

    float dist       = std::numeric_limits<float>::max();
    float dist_plane = std::numeric_limits<float>::max();
    float dist_best  = std::numeric_limits<float>::max();
    refIdx_t node_best, node_bro;
    
    std::stack<start_end> buffer;
    refIdx_t node_start, node_end;
    buffer.emplace(start_end(root, _search(point, root)));

    while (!buffer.empty())
    {
        std::tie(node_start, node_end) = buffer.top();
        buffer.pop();

        // back trace until to the starting node
        while (true)
        {
            const KdNode& node_current = kdNodes[node_end];
            
            // update if current node is better
            dist = distance<dim>(point, coordinates + numDimensions * node_current.tuple);
            if (dist < dist_best)
            {
                dist_best = dist;
                node_best = node_end;
            }

            if (node_end != node_start)
            {
                node_bro = node_current.brother;
                if (node_bro >= 0)
                {
                    // if intersect with plane, search another branch
                    dist_plane = distance_plane<dim>(point, node_current.parent);
                    if (dist_plane < dist_best)
                    {
                        buffer.emplace(start_end(node_bro, _search(point, node_bro)));
                    }
                }

                // back trace
                node_end = node_current.parent;
            }
            else break;
        }
    }

    *out_ = int64_t(kdNodes[node_best].tuple);
}





template<int N>
struct WorkerNearest
{
    static void work(TorchKDTree* tree_ptr, float* points_ptr, int64_t* raw_ptr, int numDimensions, int numQuery)
    {
        #pragma omp parallel for
        for (sint i = 0; i < numQuery; ++i) tree_ptr->_search_nearest<N>(points_ptr + i * numDimensions, raw_ptr + i);
    }
};

template<int N>
struct WorkerCUDANearest
{
    static void work(TorchKDTree* tree_ptr, float* points_ptr, int64_t* raw_ptr, int numQuery)
    {
        tree_ptr->device->Search_nearest<N>(points_ptr, raw_ptr, numQuery);
    }
};

torch::Tensor TorchKDTree::search_nearest(torch::Tensor points)
{
    CHECK_CONTIGUOUS(points);
    CHECK_FLOAT32(points);
    TORCH_CHECK(points.size(1) == numDimensions, "dimensions mismatch");
    sint numQuery = points.size(0);

    float* points_ptr = points.data_ptr<float>();
    torch::Tensor indices_tensor;

    if (is_cuda)
    {
        indices_tensor = torch::zeros({numQuery}, torch::TensorOptions()
                                                    .dtype(torch::kInt64)
                                                    .device(torch::kCUDA, device->getDevice()));
        int64_t* raw_ptr = indices_tensor.data_ptr<int64_t>();
        
        Dispatcher<WorkerCUDANearest>::dispatch(numDimensions,
                                                this, points_ptr, raw_ptr, numQuery);
    }
    else
    {
        indices_tensor = torch::zeros({numQuery}, torch::kInt64);
        int64_t* raw_ptr = indices_tensor.data_ptr<int64_t>();
        
        Dispatcher<WorkerNearest>::dispatch(numDimensions,
                                            this, points_ptr, raw_ptr, numDimensions, numQuery);
    }

    return indices_tensor;
}

#endif