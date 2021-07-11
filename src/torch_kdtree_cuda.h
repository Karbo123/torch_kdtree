
struct NodeCoordIndices
{
    refIdx_t node_index;
    refIdx_t coord_index;
};

NodeCoordIndices* d_index_temp = nullptr; // node index for temp storing, size=(num_of_points, )
NodeCoordIndices* d_index_down = nullptr; // node index for searching down, size=(num_of_points, )
NodeCoordIndices* d_index_up   = nullptr; // node index for back tracing, size=(num_of_points, )

sint* d_num_temp = nullptr; // num of temp, size=(1, )
sint* d_num_down = nullptr; // num of down searching, size=(1, )
sint* d_num_up   = nullptr; // num of up searching, size=(1, )

// https://blog.csdn.net/FreeeLinux/article/details/52075018
refIdx_t* d_queue = nullptr; // the queues, size=(num_of_points, CUDA_QUEUE_MAX), row-major
sint* d_queue_frontend = nullptr; // front and end index of queue, size=(num_of_points, 2)
#define CUDA_QUEUE_MAX (8) // the assumed max size of one queue


void init(sint num_of_points)
{
    // TODO NOTE do not cudaMalloc frequently @@@@@@@@@@@@@@@@@@@@@
    checkCudaErrors(cudaMalloc((void**)&d_index_temp, sizeof(NodeCoordIndices) * num_of_points));
    checkCudaErrors(cudaMalloc((void**)&d_index_down, sizeof(NodeCoordIndices) * num_of_points));
    checkCudaErrors(cudaMalloc((void**)&d_index_up, sizeof(NodeCoordIndices) * num_of_points));
    checkCudaErrors(cudaMalloc((void**)&d_num_temp, sizeof(sint)));
    checkCudaErrors(cudaMalloc((void**)&d_num_down, sizeof(sint)));
    checkCudaErrors(cudaMalloc((void**)&d_num_up, sizeof(sint)));
    checkCudaErrors(cudaMalloc((void**)&d_queue, sizeof(refIdx_t) * num_of_points * CUDA_QUEUE_MAX));
    checkCudaErrors(cudaMalloc((void**)&d_queue_frontend, sizeof(sint) * num_of_points * 2));
}

void desy()
{
    if (d_index_temp != nullptr)
        checkCudaErrors(cudaFree(d_index_temp));
    if (d_index_down != nullptr)
        checkCudaErrors(cudaFree(d_index_down));
    if (d_index_up != nullptr)
        checkCudaErrors(cudaFree(d_index_up));
    if (d_num_temp != nullptr)
        checkCudaErrors(cudaFree(d_num_temp));
    if (d_num_down != nullptr)
        checkCudaErrors(cudaFree(d_num_down));
    if (d_num_up != nullptr)
        checkCudaErrors(cudaFree(d_num_up));
    if (d_queue != nullptr)
        checkCudaErrors(cudaFree(d_queue));
    if (d_queue_frontend != nullptr)
        checkCudaErrors(cudaFree(d_queue_frontend));
}


// e.g. prepare d_index_down
__global__ void cuInitSearch(sint num_of_points, 
                             NodeCoordIndices* d_index_down,  refIdx_t root_index, 
                             sint* d_queue_frontend
                             )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_of_points)
    {
        d_index_down[tid].node_index = root_index;
        d_index_down[tid].coord_index = tid;
        d_queue_frontend[tid * 2 + 0] = -1; // front
        d_queue_frontend[tid * 2 + 1] = -1; // end
    }
}


// make one step to search down, and update to temp
__global__ void cuOneStepSearchDown()
{

}


// make one step to search up, and update to temp
__global__ void cuOneStepSearchUp()
{

}

