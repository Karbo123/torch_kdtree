#ifndef TORCH_KDTREE_CUDA_STACK_H_
#define TORCH_KDTREE_CUDA_STACK_H_

namespace stack_func // device functions for operating stack
{
	template<int stack_max>
	__device__ __forceinline__ bool is_full(sint* d_stack_back, sint point_index)
	{
		return d_stack_back[point_index] == stack_max - 1;			   
	}

	__device__ __forceinline__ bool is_empty(sint* d_stack_back, sint point_index)
	{
		return d_stack_back[point_index] == -1;   
	}

	template<int stack_max>
	__device__ __forceinline__ bool push(StartEndIndices* d_stack, sint* d_stack_back, sint point_index,
										 StartEndIndices* item)
	{
		if (d_stack_back[point_index] == stack_max - 1) return false; // fail to push back
		d_stack[stack_max * point_index + (++d_stack_back[point_index])] = *item;
		return true;
	}

	template<int stack_max>
	__device__ __forceinline__ bool poptop(StartEndIndices* d_stack, sint* d_stack_back, sint point_index,
										   StartEndIndices* item_out)
	{
		if (d_stack_back[point_index] == -1) return false; // fail to read value
		*item_out = d_stack[stack_max * point_index + (d_stack_back[point_index]--)];
		return true;
	}
};





// make one step to search up, and update to temp
template<int stack_max>
__global__ void cuLoadFromStack(StartEndIndices* d_stack, sint* d_stack_back, sint num_of_points,
								CoordStartEndIndices* d_index_up, sint* d_num_up)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_of_points)
    {
		StartEndIndices item;
	 	bool success = stack_func::poptop<stack_max>(d_stack, d_stack_back, tid, &item);
		if (!success) return; // for this point, stack is empty, ignore this point

		int place_index = atomicAdd(d_num_up, 1); // put here
		CoordStartEndIndices cont;
		cont.coord_index = tid;
		cont.start_index = item.start_index;
		cont.end_index   = item.end_index;
		
		d_index_up[place_index] = cont;
	}
}


// get num of empty in stack
__global__ void cuEmptyNum(sint* d_stack_back, sint num_of_points,
						   int* empty_num_out)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_of_points)
    {
		bool is_empty = stack_func::is_empty(d_stack_back, tid);
		atomicAdd(empty_num_out, (int)is_empty);
	}
}


#endif