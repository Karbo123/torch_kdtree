#ifndef TORCH_KDTREE_CUDA_STACK_H_
#define TORCH_KDTREE_CUDA_STACK_H_

#include "prefix_sum.h"

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

	template<int stack_max>
	__device__ __forceinline__ int popall(StartEndIndices* d_stack, sint* d_stack_back, sint point_index,
										  StartEndIndices** item_out)
	{
		int num = d_stack_back[point_index] + 1;
		if (num == 0) *item_out = nullptr;
		else *item_out = d_stack + stack_max * point_index;
		d_stack_back[point_index] = -1;
		return num;
	}
};





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



__global__ void cuFillIndividualNum(sint* d_stack_back, sint num_of_points,
								    sint* d_num_indi)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_of_points)
    {
		d_num_indi[tid] = d_stack_back[tid] + 1;
	}
}


__global__ void cuAddArrays(sint* data1, sint* data2, sint size)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) data1[tid] += data2[tid];
}

__global__ void cuFindFillIndex(sint* d_prefixsum, sint num_of_points, sint limit_max, sint* d_index, sint* d_num_to_fill)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_of_points)
    {
		if ( d_prefixsum[tid] < limit_max && d_prefixsum[tid] > ((tid==0) ? (-1) : d_prefixsum[tid-1]) )
		{
			int index = atomicAdd(d_num_to_fill, 1);
			d_index[index] = tid;
		}
	}
}

template<int stack_max>
__global__ void cuLoadManyFromStack(StartEndIndices* d_stack, sint* d_stack_back, 
								    sint* d_index, sint num_to_fill, sint* d_prefixsum,
							    	CoordStartEndIndices* d_index_up, sint* d_num_up)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_to_fill)
    {
		const int ind = d_index[tid]; // index of point to be loaded
		const int num = d_prefixsum[ind] - ( (ind==0) ? 0 : d_prefixsum[ind-1] ); // num to load
		
		int place_index = atomicAdd(d_num_up, num); // put here

		StartEndIndices* load_ptr = nullptr;
	 	int pop_num = stack_func::popall<stack_max>(d_stack, d_stack_back, ind, &load_ptr);
		if (pop_num != num) printf("not match error! (expected %d, but pop %d) (d_prefixsum[ind] = %d, d_prefixsum[ind-1] = %d, tid=%d, ind=%d)\n", num, pop_num, d_prefixsum[ind], ((ind==0)?0:d_prefixsum[ind-1]), tid, ind);

		// storing items
		CoordStartEndIndices cont;
		cont.coord_index = ind;
		for (int i = 0; i < pop_num; ++i)
		{
			cont.start_index = load_ptr[i].start_index;
			cont.end_index   = load_ptr[i].end_index;
			d_index_up[place_index + i] = cont;
		}
	}
}

template<int stack_max> // load as many items as possible 
void LoadManyFromStack(StartEndIndices* d_stack, sint* d_stack_back, sint num_of_points, sint load_max,
					   CoordStartEndIndices* d_index_up, sint* d_num_up,
					   sint* d_num_indi, cudaStream_t stream)
{
	// load number
	const int total_num = num_of_points;
	const int thread_num = std::min(numThreads, total_num);
	const int block_num = int(std::ceil(total_num / float(thread_num)));
	cuFillIndividualNum<<<block_num, thread_num, 0, stream>>>(d_stack_back, num_of_points, d_num_indi);
	checkCudaErrors(cudaGetLastError());

	// compute prefix sum
	sint* out_ptr = d_num_indi + num_of_points; // starts from zero
	prefix_sum::blelloch(d_num_indi, num_of_points, out_ptr);
	// add offset (make it not starting from zero)
	cuAddArrays<<<block_num, thread_num, 0, stream>>>(out_ptr, d_num_indi, num_of_points);
	checkCudaErrors(cudaGetLastError());

	// {
	// 	sint* prefixsum = new sint[num_of_points];
	// 	sint* elem = new sint[num_of_points];
	// 	checkCudaErrors(cudaMemcpyAsync(prefixsum, out_ptr, sizeof(sint) * num_of_points, cudaMemcpyDeviceToHost, stream));
	// 	checkCudaErrors(cudaMemcpyAsync(elem, d_num_indi, sizeof(sint) * num_of_points, cudaMemcpyDeviceToHost, stream));

	// 	for (int i = 100000; i < 100000 + 100; ++i)
	// 	{
	// 		cout << "i = " << i << " | " << elem[i] << " ==> " << prefixsum[i] << endl;
	// 	}


	// 	for (int i = 1; i < num_of_points; ++i)
	// 	{
	// 		if (prefixsum[i] < prefixsum[i - 1])
	// 		{
	// 			printf("not increasing! at index: %d", i);
	// 		}
	// 	}

	// 	delete[] prefixsum;
	// 	delete[] elem;
	// }

	int num_zero = 0, num_to_fill = 0;
	sint* d_num_to_fill = d_num_indi + 2 * num_of_points;
	checkCudaErrors(cudaMemcpyAsync(d_num_to_fill, &num_zero, sizeof(sint), cudaMemcpyHostToDevice, stream));
	cuFindFillIndex<<<block_num, thread_num, 0, stream>>>(out_ptr, num_of_points, load_max, d_num_indi, d_num_to_fill);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpyAsync(&num_to_fill, d_num_to_fill, sizeof(sint), cudaMemcpyDeviceToHost, stream)); 

	// {
		
	// 	sint* index = new sint[num_to_fill];
	// 	checkCudaErrors(cudaMemcpyAsync(index, d_num_indi, sizeof(sint) * num_to_fill, cudaMemcpyDeviceToHost, stream));

	// 	for (int i = 0; i < 1000; ++i)
	// 	{
	// 		cout << "index[" << i << "] = " << index[i] << endl;
	// 	}

	// 	std::sort(index, index + num_to_fill);
	// 	int size_unique = std::unique(index, index + num_to_fill) - index;
	// 	cout << "num_to_fill = " << num_to_fill << endl;
	// 	cout << "uniqued_arr size = " << size_unique << endl;

	// 	delete[] index;
	// }

	{
		const int total_num = num_to_fill;
		const int thread_num = std::min(numThreads, total_num);
		const int block_num = int(std::ceil(total_num / float(thread_num)));
		cuLoadManyFromStack <stack_max> <<<block_num, thread_num, 0, stream>>>(d_stack, d_stack_back, 
																			   d_num_indi, num_to_fill, out_ptr,
																			   d_index_up, d_num_up);
		checkCudaErrors(cudaGetLastError());
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