
#ifndef TORCH_KDTREE_KNN_H_
#define TORCH_KDTREE_KNN_H_

// search for a single point
template<int dim>
void TorchKDTree::_search_knn(const float* point, sint k, int64_t* out_)
{
    using start_end = std::tuple<refIdx_t, refIdx_t>;
    using dist_node = std::tuple<float, refIdx_t>;

    float dist       = std::numeric_limits<float>::max();
    float dist_plane = std::numeric_limits<float>::max();
    // float dist_best  = std::numeric_limits<float>::max();
    // refIdx_t node_best;
    refIdx_t node_bro;

    // create a queue
    const auto _init_cont = std::vector<dist_node>(k, dist_node(std::numeric_limits<float>::max(), -1));
    auto buffer = std::priority_queue<dist_node, std::vector<dist_node>, 
                                [](const dist_node& a, const dist_node& b)->bool{return a.get<0>() < b.get<0>();}>\
                                (_init_cont.begin(), _init_cont.end());
    
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
            dist = distance<dim>(point, coordinates_float + dim * kdNodes[node_end].tuple);
            if (dist < buffer.top()) // is smaller
            {
                buffer.pop(); // remove the largest elem
                buffer.emplace(start_end(dist, node_end)); // record the best
            }

            if (node_end != node_start)
            {
                node_bro = kdNodes[node_end].brother;
                if (node_bro >= 0)
                {
                    // if intersect with plane, search another branch
                    dist_plane = distance_plane<dim>(point, kdNodes[node_end].parent);
                    if (dist_plane < dist_best) // TODO @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
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

    *out_ = int64_t(kdNodes[node_best].tuple);
}



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
        
        if (numDimensions > 8 && 
            numDimensions != 16 && 
            numDimensions != 32)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_knn<0>(points_ptr + i * numDimensions, raw_ptr + i * k);
        }
        else if (numDimensions == 1)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_knn<1>(points_ptr + i * numDimensions, raw_ptr + i * k);
        }
        else if (numDimensions == 2)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_knn<2>(points_ptr + i * numDimensions, raw_ptr + i * k);
        }
        else if (numDimensions == 3)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_knn<3>(points_ptr + i * numDimensions, raw_ptr + i * k);
        }
        else if (numDimensions == 4)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_knn<4>(points_ptr + i * numDimensions, raw_ptr + i * k);
        }
        else if (numDimensions == 5)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_knn<5>(points_ptr + i * numDimensions, raw_ptr + i * k);
        }
        else if (numDimensions == 6)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_knn<6>(points_ptr + i * numDimensions, raw_ptr + i * k);
        }
        else if (numDimensions == 7)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_knn<7>(points_ptr + i * numDimensions, raw_ptr + i * k);
        }
        else if (numDimensions == 8)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_knn<8>(points_ptr + i * numDimensions, raw_ptr + i * k);
        }
        else if (numDimensions == 16)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_knn<16>(points_ptr + i * numDimensions, raw_ptr + i * k);
        }
        else if (numDimensions == 32)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_knn<32>(points_ptr + i * numDimensions, raw_ptr + i * k);
        }
    }

    return indices_tensor;
}

#endif