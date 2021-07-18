/*
	Codes are modified from: https://github.com/mark-poscablo/gpu-prefix-sum
*/

#ifndef PREFIX_SUM_H_
#define PREFIX_SUM_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

namespace prefix_sum
{
	#define MAX_BLOCK_SZ 1024
	#define NUM_BANKS 32
	#define LOG_NUM_BANKS 5

	#ifdef ZERO_BANK_CONFLICTS
	#define CONFLICT_FREE_OFFSET(n) \
		((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
	#else
	#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
	#endif

	template<typename DataType>
	__global__ void gpu_add_block_sums(DataType* const d_out,
									const DataType* const d_in,
									DataType* const d_block_sums,
									const size_t numElems)
	{
		auto d_block_sum_val = d_block_sums[blockIdx.x];

		// Simple implementation's performance is not significantly (if at all)
		//  better than previous verbose implementation
		const size_t cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
		if (cpy_idx < numElems)
		{
			d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
			if (cpy_idx + blockDim.x < numElems)
				d_out[cpy_idx + blockDim.x] = d_in[cpy_idx + blockDim.x] + d_block_sum_val;
		}
	}

	// Modified version of Mark Harris' implementation of the Blelloch scan
	//  according to https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf
	template<typename DataType>
	__global__ void gpu_prescan(DataType* const d_out,
								const DataType* const d_in,
								DataType* const d_block_sums,
								const size_t len,
								const size_t shmem_sz,
								const size_t max_elems_per_block)
	{
		// Allocated on invocation
		extern __shared__ DataType s_out[];

		int thid = threadIdx.x;
		int ai = thid;
		int bi = thid + blockDim.x;

		// Zero out the shared memory
		// Helpful especially when input size is not power of two
		s_out[thid] = 0;
		s_out[thid + blockDim.x] = 0;
		// If CONFLICT_FREE_OFFSET is used, shared memory
		//  must be a few more than 2 * blockDim.x
		if (thid + max_elems_per_block < shmem_sz)
			s_out[thid + max_elems_per_block] = 0;

		__syncthreads();
		
		// Copy d_in to shared memory
		// Note that d_in's elements are scattered into shared memory
		//  in light of avoiding bank conflicts
		const size_t cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
		if (cpy_idx < len)
		{
			s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
			if (cpy_idx + blockDim.x < len)
				s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
		}

		// For both upsweep and downsweep:
		// Sequential indices with conflict free padding
		//  Amount of padding = target index / num banks
		//  This "shifts" the target indices by one every multiple
		//   of the num banks
		// offset controls the stride and starting index of 
		//  target elems at every iteration
		// d just controls which threads are active
		// Sweeps are pivoted on the last element of shared memory

		// Upsweep/Reduce step
		int offset = 1;
		for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
		{
			__syncthreads();

			if (thid < d)
			{
				int ai = offset * ((thid << 1) + 1) - 1;
				int bi = offset * ((thid << 1) + 2) - 1;
				ai += CONFLICT_FREE_OFFSET(ai);
				bi += CONFLICT_FREE_OFFSET(bi);

				s_out[bi] += s_out[ai];
			}
			offset <<= 1;
		}

		// Save the total sum on the global block sums array
		// Then clear the last element on the shared memory
		if (thid == 0) 
		{ 
			d_block_sums[blockIdx.x] = s_out[max_elems_per_block - 1 
				+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
			s_out[max_elems_per_block - 1 
				+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
		}

		// Downsweep step
		for (int d = 1; d < max_elems_per_block; d <<= 1)
		{
			offset >>= 1;
			__syncthreads();

			if (thid < d)
			{
				int ai = offset * ((thid << 1) + 1) - 1;
				int bi = offset * ((thid << 1) + 2) - 1;
				ai += CONFLICT_FREE_OFFSET(ai);
				bi += CONFLICT_FREE_OFFSET(bi);

				DataType temp = s_out[ai];
				s_out[ai] = s_out[bi];
				s_out[bi] += temp;
			}
		}
		__syncthreads();

		// Copy contents of shared memory to global memory
		if (cpy_idx < len)
		{
			d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
			if (cpy_idx + blockDim.x < len)
				d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
		}
	}


	template<typename DataType>
	void blelloch(const DataType* const d_in,
				  const size_t numElems,
				  DataType* const d_out)
	{
		// Zero out d_out
		checkCudaErrors(cudaMemset(d_out, 0, numElems * sizeof(DataType)));

		// Set up number of threads and blocks
		
		size_t block_sz = MAX_BLOCK_SZ / 2;
		size_t max_elems_per_block = 2 * block_sz; // due to binary tree nature of algorithm

		// If input size is not power of two, the remainder will still need a whole block
		// Thus, number of blocks must be the ceiling of input size / max elems that a block can handle
		//unsigned int grid_sz = (unsigned int) std::ceil((double) numElems / (double) max_elems_per_block);
		// UPDATE: Instead of using ceiling and risking miscalculation due to precision, just automatically  
		//  add 1 to the grid size when the input size cannot be divided cleanly by the block's capacity
		size_t grid_sz = numElems / max_elems_per_block;
		// Take advantage of the fact that integer division drops the decimals
		if (numElems % max_elems_per_block != 0) 
			grid_sz += 1;

		// Conflict free padding requires that shared memory be more than 2 * block_sz
		size_t shmem_sz = max_elems_per_block + ((max_elems_per_block - 1) >> LOG_NUM_BANKS);

		// Allocate memory for array of total sums produced by each block
		// Array length must be the same as number of blocks
		DataType* d_block_sums;
		checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(DataType) * grid_sz));
		checkCudaErrors(cudaMemset(d_block_sums, 0, sizeof(DataType) * grid_sz));

		// Sum scan data allocated to each block
		
		gpu_prescan<<<grid_sz, block_sz, sizeof(DataType) * shmem_sz>>>(d_out, 
																		d_in, 
																		d_block_sums, 
																		numElems, 
																		shmem_sz,
																		max_elems_per_block);

		// Sum scan total sums produced by each block
		// Use basic implementation if number of total sums is <= 2 * block_sz
		//  (This requires only one block to do the scan)
		if (grid_sz <= max_elems_per_block)
		{
			DataType* d_dummy_blocks_sums;
			checkCudaErrors(cudaMalloc(&d_dummy_blocks_sums, sizeof(DataType)));
			checkCudaErrors(cudaMemset(d_dummy_blocks_sums, 0, sizeof(DataType)));

			gpu_prescan<<<1, block_sz, sizeof(DataType) * shmem_sz>>>(d_block_sums, 
																	d_block_sums, 
																	d_dummy_blocks_sums, 
																	grid_sz, 
																	shmem_sz,
																	max_elems_per_block);
			checkCudaErrors(cudaFree(d_dummy_blocks_sums));
		}
		// Else, recurse on this same function as you'll need the full-blown scan
		//  for the block sums
		else
		{
			DataType* d_in_block_sums;
			checkCudaErrors(cudaMalloc(&d_in_block_sums, sizeof(DataType) * grid_sz));
			checkCudaErrors(cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(DataType) * grid_sz, cudaMemcpyDeviceToDevice));
			blelloch(d_in_block_sums, grid_sz, d_block_sums);
			checkCudaErrors(cudaFree(d_in_block_sums));
		}

		// Add each block's total sum to its scan output
		// in order to get the final, global scanned array
		gpu_add_block_sums<DataType> <<<grid_sz, block_sz>>>(d_out, d_out, d_block_sums, numElems);

		checkCudaErrors(cudaFree(d_block_sums));
	}
};

#endif