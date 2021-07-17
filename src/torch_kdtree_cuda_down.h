#ifndef TORCH_KDTREE_CUDA_DOWN_H_
#define TORCH_KDTREE_CUDA_DOWN_H_


// make one step to search down, and update to temp
template<int stack_max>
__global__ void cuOneStepSearchDown(StartEndIndices* d_stack, sint* d_stack_back, 
									KdNode* d_kdNodes, 
									CoordStartEndIndices* d_index_down, sint* d_num_down,
									CoordStartEndIndices* d_index_temp, sint* d_num_temp,
									const float* d_query, const float* d_coord, sint numDimensions
								)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < *d_num_down)
    {
		const KdNode& node_current = d_kdNodes[d_index_down[tid].end_index];

		bool has_left_node  = (node_current.ltChild >= 0);
		bool has_right_node = (node_current.gtChild >= 0);
		
		if (has_left_node || has_right_node) // has child, go down
		{
			int place_index = atomicAdd(d_num_temp, 1); // put here
			d_index_temp[place_index].coord_index = d_index_down[tid].coord_index;
			d_index_temp[place_index].start_index = d_index_down[tid].start_index;

			if (has_left_node && has_right_node)
			{
				float val      = d_query[node_current.split_dim + numDimensions * d_index_down[tid].coord_index];
				float val_node = d_coord[node_current.split_dim + numDimensions * node_current.tuple];
				if (val < val_node) d_index_temp[place_index].end_index = node_current.ltChild;
				else d_index_temp[place_index].end_index = node_current.gtChild;
			}
			else if (has_left_node) d_index_temp[place_index].end_index = node_current.ltChild;
			else d_index_temp[place_index].end_index = node_current.gtChild;
		} else
		{
			// push in stack
			StartEndIndices item;
			refIdx_t coord_index = d_index_down[tid].coord_index;
			item.start_index     = d_index_down[tid].start_index;
			item.end_index       = d_index_down[tid].end_index;
			bool success = stack_func::push<stack_max>(d_stack, d_stack_back, coord_index, &item);
			if (!success) printf("stack is full, cannot push anymore!");
		}
	}
}


void Gpu::SearchDown(const float* d_query)
{
	const int numDimensions = dimen;
	sint num_zero = 0;
	while (true) // all reach the leaf
	{
		sint num_to_down = 0;
		checkCudaErrors(cudaMemcpyAsync(d_num_temp, &num_zero, sizeof(sint), cudaMemcpyHostToDevice, stream));
		checkCudaErrors(cudaMemcpyAsync(&num_to_down, d_num_down, sizeof(sint), cudaMemcpyDeviceToHost, stream));

		if (num_to_down == 0) break; // no longer to search down

		const int total_num = num_to_down;
		const int thread_num = std::min(numThreads, total_num);
		const int block_num = int(std::ceil(total_num / float(thread_num)));
		cuOneStepSearchDown <CUDA_STACK_MAX> <<<block_num, thread_num, 0, stream>>> (d_stack, d_stack_back, d_kdNodes, d_index_down, d_num_down, d_index_temp, d_num_temp, d_query, d_coord, numDimensions);
		checkCudaErrors(cudaGetLastError());

		std::swap(d_index_down, d_index_temp);
		std::swap(d_num_down, d_num_temp);
	}
}


#endif