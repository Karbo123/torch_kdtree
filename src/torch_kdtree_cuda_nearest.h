#ifndef TORCH_KDTREE_CUDA_NEAREST_H_
#define TORCH_KDTREE_CUDA_NEAREST_H_

// make one step to search up, and update to temp
template<int dim>
__global__ void cuOneStepSearchUp_nearest(CoordStartEndIndices* d_index_up, sint* d_num_up,
										  CoordStartEndIndices* d_index_temp, sint* d_num_temp, // num init to zero
										  CoordStartEndIndices* d_index_down, sint* d_num_down, // num init to zero
                                          const float* d_query, const float* d_coord, const KdNode* d_kdNodes,
                                          ResultNearest* d_result_buffer
                                        )
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < *d_num_up)
    {
		const CoordStartEndIndices& indices = d_index_up[tid];
		const sint coord_index = indices.coord_index;
        const float* point = d_query + dim * coord_index;

        float dist = distance<dim>(point, d_coord + dim * d_kdNodes[indices.end_index].tuple);
        ResultNearest& result = d_result_buffer[coord_index];
        if (dist < result.dist)
        {
            result.dist = dist;
            result.best_index = indices.end_index;
        }

        if (indices.end_index == indices.start_index) return; // if reach the top, then the search is done
		
		int place_index = atomicAdd(d_num_temp, 1); // put here
		CoordStartEndIndices item;
		item.coord_index = coord_index;
		item.start_index = indices.start_index;
		item.end_index   = d_kdNodes[indices.end_index].parent; // continue to back trace
		d_index_temp[place_index] = item;

        refIdx_t node_bro = d_kdNodes[indices.end_index].brother;
        if (node_bro >= 0)
        {
            float dist_plane = distance_plane<dim>(d_kdNodes, d_coord, point, d_kdNodes[indices.end_index].parent);
            if (dist_plane < result.dist)
            {
				int place_index = atomicAdd(d_num_down, 1); // put here, and wait to search down
				CoordStartEndIndices item;
				item.coord_index = coord_index;
				item.start_index = node_bro;
				item.end_index   = node_bro;
				d_index_down[place_index] = item;
            }
        }
	}
}


template<int dim>
void Gpu::OneStepSearchUp_nearest(const float* d_query)
{
	sint num_zero = 0;
	checkCudaErrors(cudaMemcpyAsync(d_num_down, &num_zero, sizeof(sint), cudaMemcpyHostToDevice, stream));

	sint num_up = 0;
	checkCudaErrors(cudaMemcpyAsync(&num_up, d_num_up, sizeof(sint), cudaMemcpyDeviceToHost, stream));
	if (num_up == 0) // empty, load from stack
	{
		const int total_num = num_of_points;
		const int thread_num = std::min(numThreads, total_num);
		const int block_num = int(std::ceil(total_num / float(thread_num)));
		cuLoadFromStack <CUDA_STACK_MAX> <<<block_num, thread_num, 0, stream>>> (d_stack, d_stack_back, num_of_points, d_index_up, d_num_up);
		checkCudaErrors(cudaGetLastError());
		// load num to host
		checkCudaErrors(cudaMemcpyAsync(&num_up, d_num_up, sizeof(sint), cudaMemcpyDeviceToHost, stream));
	}
	
	// make one step to search up
	checkCudaErrors(cudaMemcpyAsync(d_num_temp, &num_zero, sizeof(sint), cudaMemcpyHostToDevice, stream));
	const int total_num = num_up;
	const int thread_num = std::min(numThreads, total_num);
	const int block_num = int(std::ceil(total_num / float(thread_num)));
	cuOneStepSearchUp_nearest<dim> <<<block_num, thread_num, 0, stream>>> (d_index_up, d_num_up,
																		   d_index_temp, d_num_temp,
																		   d_index_down, d_num_down,
																		   d_query, d_coord, d_kdNodes,
																		   (ResultNearest*) d_result_buffer
																		);
	checkCudaErrors(cudaGetLastError());
	std::swap(d_index_up, d_index_temp);
	std::swap(d_num_up, d_num_temp);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

// __global__
// TODO naive method @@@@@@@@@@@@@@@@@@@@@@@@@



/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void CopyResult_nearest(const ResultNearest* result, const KdNode* d_kdNodes, const int num_of_points, int64_t* index_out)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < num_of_points)
	{
		index_out[tid] = int64_t(d_kdNodes[result[tid].best_index].tuple); // it should be index of point, not kdnode
	}
}

__global__ void InitResult_nearest(ResultNearest* result, const int num_of_points)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < num_of_points)
	{
		result[tid].best_index = -1;
		result[tid].dist = FLT_MAX; // very large distance
	}
}



template<int dim>
void Gpu::Search_nearest(const float* d_query, int64_t* index_out, const int _num_of_points)
{
	InitSearch(_num_of_points, Nearest);

	// init result
	const int total_num = num_of_points;
	const int thread_num = std::min(numThreads, total_num);
	const int block_num = int(std::ceil(total_num / float(thread_num)));
	InitResult_nearest<<<block_num, thread_num, 0, stream>>>((ResultNearest*) d_result_buffer, num_of_points);
	checkCudaErrors(cudaGetLastError());

	int ____COUNTER____ = 0;
	Timer timer;
	Timer timer_sum;
	while (true)
	{
		SearchDown(d_query);
		OneStepSearchUp_nearest<dim>(d_query);

		// empty num
		int num_empty = 0;
		checkCudaErrors(cudaMemcpyAsync(d_num_empty, &num_empty, sizeof(sint), cudaMemcpyHostToDevice, stream));
		cuEmptyNum<<<block_num, thread_num, 0, stream>>>(d_stack_back, num_of_points, d_num_empty);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaMemcpyAsync(&num_empty, d_num_empty, sizeof(sint), cudaMemcpyDeviceToHost, stream));
		
		// down num
		int num_down = 0;
		checkCudaErrors(cudaMemcpyAsync(&num_down, d_num_down, sizeof(sint), cudaMemcpyDeviceToHost, stream));

		// num up
		int num_up = 0;
		checkCudaErrors(cudaMemcpyAsync(&num_up, d_num_up, sizeof(sint), cudaMemcpyDeviceToHost, stream));

		checkCudaErrors(cudaDeviceSynchronize());
		double time_sum_ms = timer_sum.elapsed() * 1000;
		double time_ms = timer.elapsed() * 1000; timer.reset();
		cout << "____COUNTER____ = " << ____COUNTER____ << endl;
		cout << "[DEBUG] num_empty = " << num_empty << "; left = " << num_of_points - num_empty << endl;
		cout << "[DEBUG] num_down = " << num_down << endl;
		cout << "[DEBUG] num_up = " << num_up << endl;
		cout << "[DEBUG] time = " << time_ms << "ms" << endl;
		cout << "[DEBUG] time_sum = " << time_sum_ms << "ms" << endl;
		____COUNTER____++;

		if (num_empty == num_of_points // becomes empty
			&& num_down == 0 // no need to search down
			&& num_up == 0 // no need to search up
		) break; // all done
	}

	// copy result index
	CopyResult_nearest<<<block_num, thread_num, 0, stream>>>((ResultNearest*) d_result_buffer, d_kdNodes, num_of_points, index_out);
	checkCudaErrors(cudaGetLastError());
}


#endif