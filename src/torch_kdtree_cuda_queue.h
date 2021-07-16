#ifndef TORCH_KDTREE_CUDA_QUEUE_H_
#define TORCH_KDTREE_CUDA_QUEUE_H_

namespace queue_func // device functions for operating queue
{
	template<int queue_max>
	__device__ __forceinline__ bool queue_is_full(FrontEndIndices* d_queue_frontend, sint point_index)
	{
		const FrontEndIndices& indices = d_queue_frontend[point_index];
		return (indices.end_index + 1) % queue_max == indices.front_index;			   
	}

	__device__ __forceinline__ bool queue_is_empty(FrontEndIndices* d_queue_frontend, sint point_index)
	{
		const FrontEndIndices& indices = d_queue_frontend[point_index];
		return indices.front_index == -1;   
	}

	template<int queue_max> // only read the front value
	__device__ __forceinline__ bool queue_front(StartEndIndices* d_queue, FrontEndIndices* d_queue_frontend, sint point_index,
												StartEndIndices* item_out)
	{
		if (queue_is_empty(d_queue_frontend, point_index)) return false; // fail to read value
		*item_out = d_queue[queue_max * point_index + d_queue_frontend[point_index].front_index];
		return true;
	}

	template<int queue_max>
	__device__ __forceinline__ bool queue_pushback(StartEndIndices* d_queue, FrontEndIndices* d_queue_frontend, sint point_index,
												   StartEndIndices* item)
	{
		if (queue_is_full<queue_max>(d_queue_frontend, point_index)) return false; // fail to push back
		FrontEndIndices& indices = d_queue_frontend[point_index];
		sint storage_index = 0;
		if (indices.front_index != -1 && indices.end_index != queue_max - 1) storage_index = (++indices.end_index);
		else indices.end_index = 0;
		d_queue[queue_max * point_index + storage_index] = *item;
		if (indices.front_index == -1) indices.front_index = 0;
		return true;
	}

	template<int queue_max> // also pop the value
	__device__ __forceinline__ bool queue_popfront(StartEndIndices* d_queue, FrontEndIndices* d_queue_frontend, sint point_index,
												   StartEndIndices* item_out)
	{
		if (queue_is_empty(d_queue_frontend, point_index)) return false; // fail to pop front
		FrontEndIndices& indices = d_queue_frontend[point_index];
		*item_out = d_queue[queue_max * point_index + indices.front_index]; // inplace
		if (indices.front_index == indices.end_index)
		{
			indices.front_index = -1;
			indices.end_index = -1;
		} else
		{
			indices.front_index = (indices.front_index + 1) % queue_max;
		}
		
		return true;
	}

	template<int queue_max> // only pop, not pop out value
	__device__ __forceinline__ bool queue_popfront(StartEndIndices* d_queue, FrontEndIndices* d_queue_frontend, sint point_index)
	{
		if (queue_is_empty(d_queue_frontend, point_index)) return false; // fail to pop front
		FrontEndIndices& indices = d_queue_frontend[point_index];
		if (indices.front_index == indices.end_index)
		{
			indices.front_index = -1;
			indices.end_index = -1;
		} else
		{
			indices.front_index = (indices.front_index + 1) % queue_max;
		}
		
		return true;
	}
};


// make one step to search up, and update to temp
template<int queue_max>
__global__ void cuLoadFromQueue(StartEndIndices* d_queue, FrontEndIndices* d_queue_frontend, sint num_of_points,
								CoordStartEndIndices* d_index_up, sint* d_num_up)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_of_points)
    {
		StartEndIndices item;
	 	bool success = queue_func::queue_popfront<queue_max>(d_queue, d_queue_frontend, tid, &item);
		if (!success) return; // for this point, queue is empty, ignore this point

		int place_index = atomicAdd(d_num_up, 1); // put here
		CoordStartEndIndices cont;
		cont.coord_index = tid;
		cont.start_index = item.start_index;
		cont.end_index   = item.end_index;
		
		d_index_up[place_index] = cont;
	}
}


// get num of empty in queue
__global__ void cuEmptyNum(FrontEndIndices* d_queue_frontend, sint num_of_points,
						   int* empty_num_out)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_of_points)
    {
		bool is_empty = queue_func::queue_is_empty(d_queue_frontend, tid);
		atomicAdd(empty_num_out, (int)is_empty);
	}
}


#endif