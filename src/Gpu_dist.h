

template<int dim>
__device__ __forceinline__ float distance(const float* point_a, const float* point_b)
{
    float sum = 0;
    #pragma unroll
    for (sint i = 0; i < dim; ++i)
        sum += POW2(point_a[i] - point_b[i]);
    return sum;
}


template<>
__device__ __forceinline__ float distance<3>(const float* point_a, const float* point_b)
{
    return POW2(point_a[0] - point_b[0]) + \
           POW2(point_a[1] - point_b[1]) + \
           POW2(point_a[2] - point_b[2]); 
}


template<int dim>
__device__ __forceinline__ float distance_plane(const KdNode* d_kdNodes, const float* d_coord, 
                                                const float* point, refIdx_t node)
{
    const KdNode& node_plane = d_kdNodes[node];
    return POW2(point[node_plane.split_dim] - d_coord[dim * node_plane.tuple + node_plane.split_dim]);
}
