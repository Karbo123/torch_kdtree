
// make one step to search up, and update to temp
template<int dim>
__global__ void cuOneStepSearchUp_nearest(StartEndIndices* d_queue, FrontEndIndices* d_queue_frontend,
                                          CoordStartEndIndices* d_index_up, sint* d_num_up,
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
void Gpu::SearchUp_nearest(const float* d_query)
{
	const int numDimensions = dimen;

	sint num_zero = 0;
	checkCudaErrors(cudaMemcpyAsync(d_num_down, &num_zero, sizeof(sint), cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_num_up, &num_zero, sizeof(sint), cudaMemcpyHostToDevice, stream));

	// load from queue
	const int total_num = num_of_points;
	const int thread_num = std::min(numThreads, total_num);
	const int block_num = int(std::ceil(total_num / float(thread_num)));
	cuLoadFromQueue <CUDA_QUEUE_MAX> <<<block_num, thread_num, 0, stream>>> (d_queue, d_queue_frontend, num_of_points, d_index_up, d_num_up);
	checkCudaErrors(cudaGetLastError());

	while (true)
	{
		sint num_up = 0;
		checkCudaErrors(cudaMemcpyAsync(d_num_temp, &num_up, sizeof(sint), cudaMemcpyHostToDevice, stream));
		checkCudaErrors(cudaMemcpyAsync(&num_up, d_num_up, sizeof(sint), cudaMemcpyDeviceToHost, stream));
		
		if (num_up == 0) break;

		const int total_num = num_up;
		const int thread_num = std::min(numThreads, total_num);
		const int block_num = int(std::ceil(total_num / float(thread_num)));
		cuOneStepSearchUp_nearest<dim> <<<block_num, thread_num, 0, stream>>> (d_queue, d_queue_frontend,
																	           d_index_up, d_num_up,
																	           d_index_temp, d_num_temp,
																	           d_index_down, d_num_down,
																	           d_query, d_coord, d_kdNodes,
																	           (ResultNearest*) d_result_buffer
																	        );
		checkCudaErrors(cudaGetLastError());
        
        std::swap(d_index_up, d_index_temp);
		std::swap(d_num_up, d_num_temp);
	}
}



__global__ void CopyResult_nearest(const ResultNearest* result, const int num_of_points, int64_t* index_out)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < num_of_points)
	{
		index_out[tid] = int64_t(result[tid].best_index);
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
	InitSearch(_num_of_points);

	// init result
	const int total_num = num_of_points;
	const int thread_num = std::min(numThreads, total_num);
	const int block_num = int(std::ceil(total_num / float(thread_num)));
	InitResult_nearest<<<block_num, thread_num, 0, stream>>>((ResultNearest*) d_result_buffer, num_of_points);
	checkCudaErrors(cudaGetLastError());

	int num_empty = 0;
	int num_down = 0;
	while (true)
	{
		SearchDown(d_query);
		SearchUp_nearest<dim>(d_query);
		
		cuEmptyNum<<<block_num, thread_num, 0, stream>>>(d_queue_frontend, num_of_points, d_num_empty);
		checkCudaErrors(cudaGetLastError());

		checkCudaErrors(cudaMemcpyAsync(&num_empty, d_num_empty, sizeof(sint), cudaMemcpyDeviceToHost, stream));
		checkCudaErrors(cudaMemcpyAsync(&num_down, d_num_down, sizeof(sint), cudaMemcpyDeviceToHost, stream));
		if (num_empty == num_of_points // queue becomes empty
			&& num_down == 0 // no need to search down
		) break; // all done
	}

	// copy result index
	CopyResult_nearest<<<block_num, thread_num, 0, stream>>>((ResultNearest*) d_result_buffer, num_of_points, index_out);
	checkCudaErrors(cudaGetLastError());
}

