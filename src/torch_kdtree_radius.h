
#ifndef TORCH_KDTREE_RADIUS_H_
#define TORCH_KDTREE_RADIUS_H_


// search for a single point
template<int dim>
void TorchKDTree::_search_radius(const float* point, float radius2, std::vector<sint>& out_)
{
    using start_end = std::tuple<refIdx_t, refIdx_t>;

    float dist       = std::numeric_limits<float>::max();
    float dist_plane = std::numeric_limits<float>::max();
    refIdx_t node_bro;

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
            dist = distance<dim>(point, coordinates + dim * kdNodes[node_end].tuple);
            if (dist < radius2) // exists a smaller value
            {
                out_.push_back(kdNodes[node_end].tuple);
            }

            if (node_end != node_start)
            {
                node_bro = kdNodes[node_end].brother;
                if (node_bro >= 0)
                {
                    // if intersect with plane, search another branch
                    dist_plane = distance_plane<dim>(point, kdNodes[node_end].parent);
                    if (dist_plane < radius2)
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

    // sort
    std::sort(out_.begin(), out_.end());
}



// search for a single point
template<>
void TorchKDTree::_search_radius<0>(const float* point, float radius2, std::vector<sint>& out_)
{
    using start_end = std::tuple<refIdx_t, refIdx_t>;

    float dist       = std::numeric_limits<float>::max();
    float dist_plane = std::numeric_limits<float>::max();
    refIdx_t node_bro;

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
            dist = distance<0>(point, coordinates + numDimensions * kdNodes[node_end].tuple);
            if (dist < radius2) // exists a smaller value
            {
                out_.push_back(kdNodes[node_end].tuple);
            }

            if (node_end != node_start)
            {
                node_bro = kdNodes[node_end].brother;
                if (node_bro >= 0)
                {
                    // if intersect with plane, search another branch
                    dist_plane = distance_plane<0>(point, kdNodes[node_end].parent);
                    if (dist_plane < radius2)
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

    // sort
    std::sort(out_.begin(), out_.end());
}



std::tuple<torch::Tensor, torch::Tensor> TorchKDTree::search_radius(torch::Tensor points, float radius)
{
    CHECK_CONTIGUOUS(points);
    CHECK_FLOAT32(points);
    TORCH_CHECK(points.size(1) == numDimensions, "dimensions mismatch");
    sint numQuery = points.size(0);

    float* points_ptr = points.data_ptr<float>();
    torch::Tensor indices_tensor, batch_tensor;

    float radius2 = POW2(radius);

    if (is_cuda)
    {
        throw runtime_error("CUDA-KDTree cannot do searching");
    }
    else
    {
        auto vec_of_vec = std::vector<std::vector<sint>>(numQuery);

        if (numDimensions > 8 && 
            numDimensions != 16 && 
            numDimensions != 32)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_radius<0>(points_ptr + i * numDimensions, radius2, vec_of_vec[i]);
        }
        else if (numDimensions == 1)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_radius<1>(points_ptr + i * numDimensions, radius2, vec_of_vec[i]);
        }
        else if (numDimensions == 2)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_radius<2>(points_ptr + i * numDimensions, radius2, vec_of_vec[i]);
        }
        else if (numDimensions == 3)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_radius<3>(points_ptr + i * numDimensions, radius2, vec_of_vec[i]);
        }
        else if (numDimensions == 4)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_radius<4>(points_ptr + i * numDimensions, radius2, vec_of_vec[i]);
        }
        else if (numDimensions == 5)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_radius<5>(points_ptr + i * numDimensions, radius2, vec_of_vec[i]);
        }
        else if (numDimensions == 6)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_radius<6>(points_ptr + i * numDimensions, radius2, vec_of_vec[i]);
        }
        else if (numDimensions == 7)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_radius<7>(points_ptr + i * numDimensions, radius2, vec_of_vec[i]);
        }
        else if (numDimensions == 8)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_radius<8>(points_ptr + i * numDimensions, radius2, vec_of_vec[i]);
        }
        else if (numDimensions == 16)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_radius<16>(points_ptr + i * numDimensions, radius2, vec_of_vec[i]);
        }
        else if (numDimensions == 32)
        {
            #pragma omp parallel for
            for (sint i = 0; i < numQuery; ++i) _search_radius<32>(points_ptr + i * numDimensions, radius2, vec_of_vec[i]);
        }

        std::vector<sint> ind_size(numQuery);
        std::transform(vec_of_vec.begin(), vec_of_vec.end(), ind_size.begin(), [](const std::vector<sint>& vec){return sint(vec.size());});

        std::vector<sint> sum_ind_size(numQuery + 1); sum_ind_size[0] = 0;
        std::partial_sum(ind_size.begin(), ind_size.end(), sum_ind_size.begin() + 1, std::plus<sint>());
        sint sum_size = sum_ind_size.back();
        sum_ind_size.pop_back();

        indices_tensor = torch::zeros({sum_size}, torch::kInt64);
        int64_t* indices_tensor_ptr = indices_tensor.data_ptr<int64_t>();
        batch_tensor = torch::zeros({sum_size}, torch::kInt64);
        int64_t* batch_tensor_ptr = batch_tensor.data_ptr<int64_t>();

        #pragma omp parallel for
        for (sint i = 0; i < numQuery; ++i)
        {
            sint length = vec_of_vec[i].size();
            sint start_index = sum_ind_size[i];
            std::copy(vec_of_vec[i].begin(), vec_of_vec[i].end(), indices_tensor_ptr + start_index); // index
            std::fill(batch_tensor_ptr + start_index, batch_tensor_ptr + start_index + length, i);   // batch
        }
    }

    return std::make_tuple(indices_tensor, batch_tensor);
}

#endif
