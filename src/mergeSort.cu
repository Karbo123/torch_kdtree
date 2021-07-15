//
//  mergeSort.cu
//
//  Created by John Robinson on 7/15/15.
//  Copyright (c) 2015 John Robinson. All rights reserved.
/*
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSEARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */
//

/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * Merge Sort is based on 
 * "Designing efficient sorting algorithms for manycore GPUs"
 * by Nadathur Satish, Mark Harris, and Michael Garland
 * http://mgarland.org/files/papers/gpusort-ipdps09.pdf
 *
 * Victor Podlozhnyuk 09/24/2009
 */

/* The Multi-GPU Data swap is based on "
 * "Comparison Based Sorting for Systems with Multiple GPUs"
 * by van Tanasic, Lluís Vilanova, Marc Jordà, Javier Cabezas, Isaac Gelado,
      Nacho Navarro, Wen-mei Hwu
 * http://impact.crhc.illinois.edu/shared/papers/p1-tanasic.pdf
 */

/* Post Multi-GPU data swap merge is based on 
 * "GPU Merge Path - A GPU Merging Algorithm"
 *  by Oded Green, Robert McColl, David A. Bader
 ∗ http://www.cc.gatech.edu/~bader/papers/GPUMergePath-ICS2012.pdf
 */

#include <assert.h>
#include <helper_cuda.h>
#include <omp.h>
#include "Gpu.h"
#include "mergeSort_common.h"
#include <cuda_runtime.h>
#include <helper_functions.h>


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
static inline __host__ __device__ uint iDivUp(uint a, uint b)
{
	return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

static inline __host__ __device__ uint getSampleCount(uint dividend)
{
	return iDivUp(dividend, SAMPLE_STRIDE);
}

#define W (sizeof(uint) * 8)
static inline __device__ uint nextPowerOfTwo(uint x)
{
	/*
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
	 */
	return 1U << (W - __clz(x - 1));
}

template<uint sortDir> static inline __device__ uint binarySearchInclusive(uint val, uint *data, uint L, uint stride)
{
	if (L == 0)
	{
		return 0;
	}

	uint pos = 0;

	for (; stride > 0; stride >>= 1)
	{
		uint newPos = umin(pos + stride, L);

		if ((sortDir && (data[newPos - 1] <= val)) || (!sortDir && (data[newPos - 1] >= val)))
		{
			pos = newPos;
		}
	}

	return pos;
}

template<uint sortDir> static inline __device__ uint binarySearchExclusive(uint val, uint *data, uint L, uint stride)
{
	if (L == 0)
	{
		return 0;
	}

	uint pos = 0;

	for (; stride > 0; stride >>= 1)
	{
		uint newPos = umin(pos + stride, L);

		if ((sortDir && (data[newPos - 1] < val)) || (!sortDir && (data[newPos - 1] > val)))
		{
			pos = newPos;
		}
	}

	return pos;
}

// supperValCompareFirstDim performs the same function as supperValCompare.  But in the case where the calling code
// has prefetched the first dimension or component, the this takes the A and B components as L values and
// only access the array values if the first components happen to be equal.
// inputs
// ap         first compare component l value
// bp         first compare component l value
// *a         a coordinates
// *b         b coordinates
// p          index of the first

// __device__ int skc_error;

__device__ KdCoord superKeyCompareFirstDimSmpl(const KdCoord ap, const KdCoord bp, const KdCoord *a, const KdCoord *b, const sint p, const sint dim)
{
	KdCoord diff = ap - bp;
	for (sint i = 1; diff == 0 && i < dim; i++) {
		sint r = i + p;
		r = (r < dim) ? r : r - dim;
		diff = a[r] - b[r];
	}
	return diff;
}

template<uint sortDir> static inline __device__ uint binarySearchValInclusive(KdCoord coord[], KdCoord val, refIdx_t ref, KdCoord *inVal, refIdx_t *inRef, uint L, uint stride, sint p, sint dim)
{
	if (L == 0)
	{
		return 0;
	}

	uint pos = 0;

	for (; stride > 0; stride >>= 1)
	{
		uint newPos = umin(pos + stride, L);

		if (( sortDir && (superKeyCompareFirstDimSmpl(inVal[newPos - 1], val, coord+inRef[newPos - 1]*dim, coord+ref*dim, p, dim) <= 0)) ||
				(!sortDir && (superKeyCompareFirstDimSmpl(inVal[newPos - 1], val, coord+inRef[newPos - 1]*dim, coord+ref*dim, p, dim) >= 0)))
		{
			pos = newPos;
		}
	}

	return pos;
}

template<uint sortDir> static inline __device__ uint binarySearchValExclusive(KdCoord coord[], KdCoord val, refIdx_t ref, KdCoord *inVal, refIdx_t *inRef, uint L, uint stride, sint p, sint dim)
{
	if (L == 0)
	{
		return 0;
	}

	uint pos = 0;

	for (; stride > 0; stride >>= 1)
	{
		uint newPos = umin(pos + stride, L);

		if (( sortDir && (superKeyCompareFirstDimSmpl(inVal[newPos - 1], val, coord+inRef[newPos - 1]*dim, coord+ref*dim, p, dim) < 0)) ||
				(!sortDir && (superKeyCompareFirstDimSmpl(inVal[newPos - 1], val, coord+inRef[newPos - 1]*dim, coord+ref*dim, p, dim) > 0)))
		{
			pos = newPos;
		}
	}

	return pos;
}

template<uint sortDir> static inline __device__ uint binarySearchVal(bool inclusive, KdCoord coord[], KdCoord val, refIdx_t ref, KdCoord *inVal, refIdx_t *inRef, uint L, uint stride, sint p, sint dim)
{
	if (L == 0)
	{
		return 0;
	}

	uint pos = 0;

	for (; stride > 0; stride >>= 1)
	{
		uint newPos = umin(pos + stride, L);
		KdCoord comp = superKeyCompareFirstDimSmpl(inVal[newPos - 1], val, coord+inRef[newPos - 1]*dim, coord+ref*dim, p, dim);
		if ((sortDir && (comp < 0)) || (!sortDir && (comp > 0)) || (inclusive && (comp == 0)))
		{
			pos = newPos;
		}
	}

	return pos;
}



////////////////////////////////////////////////////////////////////////////////
// Bottom-level merge sort (binary search-based)
////////////////////////////////////////////////////////////////////////////////
template<uint sortDir> __global__ void mergeSortSharedKernel(
		KdCoord coord[],
		KdCoord  *d_DstVal,
		refIdx_t *d_DstRef,
		KdCoord  *d_SrcVal,
		refIdx_t *d_SrcRef,
		uint arrayLength,
		sint p,
		sint dim
)
{
	__shared__ KdCoord  s_Val[SHARED_SIZE_LIMIT];
	__shared__ refIdx_t s_Ref[SHARED_SIZE_LIMIT];

	d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	d_SrcRef += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	d_DstRef += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	s_Val[threadIdx.x +                       0] = d_SrcVal[                      0];
	s_Ref[threadIdx.x +                       0] = d_SrcRef[                      0];
	s_Val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];
	s_Ref[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcRef[(SHARED_SIZE_LIMIT / 2)];

	for (uint stride = 1; stride < arrayLength; stride <<= 1)
	{
		uint     lPos = threadIdx.x & (stride - 1);
		KdCoord  *baseVal = s_Val + 2 * (threadIdx.x - lPos);
		refIdx_t *baseRef = s_Ref + 2 * (threadIdx.x - lPos);

		__syncthreads();
		KdCoord  ValA = baseVal[lPos +      0];
		refIdx_t RefA = baseRef[lPos +      0];
		KdCoord  ValB = baseVal[lPos + stride];
		refIdx_t RefB = baseRef[lPos + stride];
		uint posA = binarySearchValExclusive<sortDir>(coord, ValA, RefA, baseVal + stride, baseRef + stride, stride, stride, p, dim) + lPos;
		uint posB = binarySearchValInclusive<sortDir>(coord, ValB, RefB, baseVal +      0, baseRef +      0, stride, stride, p, dim) + lPos;

		__syncthreads();
		baseVal[posA] = ValA;
		baseRef[posA] = RefA;
		baseVal[posB] = ValB;
		baseRef[posB] = RefB;
	}

	__syncthreads();
	d_DstVal[                      0] = s_Val[threadIdx.x +                       0];
	d_DstRef[                      0] = s_Ref[threadIdx.x +                       0];
	d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_Val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
	d_DstRef[(SHARED_SIZE_LIMIT / 2)] = s_Ref[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

void Gpu::mergeSortShared(
		KdCoord d_coords[],
		KdCoord  *d_DstVal,
		refIdx_t *d_DstRef,
		KdCoord  *d_SrcVal,
		refIdx_t *d_SrcRef,
		uint     batchSize,
		uint     arrayLength,
		uint     sortDir,
		sint      p,
		sint      dim
)
{
	if (arrayLength < 2)
	{
		return;
	}

	assert(SHARED_SIZE_LIMIT % arrayLength == 0);
	assert(((batchSize * arrayLength) % SHARED_SIZE_LIMIT) == 0);
	uint  blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
	uint threadCount = SHARED_SIZE_LIMIT / 2;

#pragma omp critical (launchLock)
{
		setDevice();
		if (sortDir)
		{
			mergeSortSharedKernel<1U><<<blockCount, threadCount, 0, stream>>>(d_coords, d_DstVal, d_DstRef, d_SrcVal, d_SrcRef, arrayLength, p, dim);
			getLastCudaError("mergeSortShared<1><<<>>> failed\n");
		}
		else
		{
			mergeSortSharedKernel<0U><<<blockCount, threadCount, 0, stream>>>(d_coords, d_DstVal, d_DstRef, d_SrcVal, d_SrcRef, arrayLength, p, dim);
			getLastCudaError("mergeSortShared<0><<<>>> failed\n");
		}
}
}



////////////////////////////////////////////////////////////////////////////////
// Merge step 1: generate sample ranks
////////////////////////////////////////////////////////////////////////////////
template<uint sortDir> __global__ void generateSampleRanksKernel(
		KdCoord coords[],
		uint     *d_RanksA,
		uint     *d_RanksB,
		KdCoord  *d_SrcVal,
		refIdx_t *d_SrcRef,
		uint     stride,
		uint     N,
		uint     threadCount,
		sint      p,
		sint      dim
)
{
	uint pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos >= threadCount)
	{
		return;
	}

	const uint           i = pos & ((stride / SAMPLE_STRIDE) - 1);
	const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
	d_SrcVal += segmentBase;
	d_SrcRef += segmentBase;
	d_RanksA += segmentBase / SAMPLE_STRIDE;
	d_RanksB += segmentBase / SAMPLE_STRIDE;

	const uint segmentElementsA = stride;
	const uint segmentElementsB = umin(stride, N - segmentBase - stride);
	const uint segmentSamplesA = getSampleCount(segmentElementsA);
	const uint segmentSamplesB = getSampleCount(segmentElementsB);

	if (i < segmentSamplesA)
	{
		d_RanksA[i] = i * SAMPLE_STRIDE;
		d_RanksB[i] = binarySearchValExclusive<sortDir>(coords,
				d_SrcVal[i * SAMPLE_STRIDE], d_SrcRef[i * SAMPLE_STRIDE],
				d_SrcVal + stride, d_SrcRef + stride,
				segmentElementsB, nextPowerOfTwo(segmentElementsB), p, dim
		);
	}

	if (i < segmentSamplesB)
	{
		d_RanksB[(stride / SAMPLE_STRIDE) + i] = i * SAMPLE_STRIDE;
		d_RanksA[(stride / SAMPLE_STRIDE) + i] = binarySearchValInclusive<sortDir>(coords,
				d_SrcVal[stride + i * SAMPLE_STRIDE], d_SrcRef[stride + i * SAMPLE_STRIDE],
				d_SrcVal + 0, d_SrcRef + 0,
				segmentElementsA, nextPowerOfTwo(segmentElementsA), p, dim
		);
	}
}

void Gpu::generateSampleRanks(
		KdCoord d_coords[],
		uint     *d_RanksA,
		uint     *d_RanksB,
		KdCoord  *d_SrcVal,
		refIdx_t *d_SrcRef,
		uint     stride,
		uint     N,
		uint     sortDir,
		sint      p,
		sint      dim
)
{
	uint lastSegmentElements = N % (2 * stride);
	uint         threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

#pragma omp critical (launchLock)
	{
		setDevice();
		if (sortDir)
		{
			generateSampleRanksKernel<1U><<<iDivUp(threadCount, 256), 256, 0, stream>>>(d_coords, d_RanksA, d_RanksB, d_SrcVal, d_SrcRef, stride, N, threadCount, p, dim);
			getLastCudaError("generateSampleRanksKernel<1U><<<>>> failed\n");
		}
		else
		{
			generateSampleRanksKernel<0U><<<iDivUp(threadCount, 256), 256, 0, stream>>>(d_coords, d_RanksA, d_RanksB, d_SrcVal, d_SrcRef, stride, N, threadCount, p, dim);
			getLastCudaError("generateSampleRanksKernel<0U><<<>>> failed\n");
		}
	}
}



////////////////////////////////////////////////////////////////////////////////
// Merge step 2: generate sample ranks and indices
////////////////////////////////////////////////////////////////////////////////
__global__ void mergeRanksAndIndicesKernel(
		uint *d_Limits,
		uint *d_Ranks,
		uint stride,
		uint N,
		uint threadCount
)
{
	uint pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos >= threadCount)
	{
		return;
	}

	const uint           i = pos & ((stride / SAMPLE_STRIDE) - 1);
	const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
	d_Ranks  += (pos - i) * 2;
	d_Limits += (pos - i) * 2;

	const uint segmentElementsA = stride;
	const uint segmentElementsB = umin(stride, N - segmentBase - stride);
	const uint segmentSamplesA = getSampleCount(segmentElementsA);
	const uint segmentSamplesB = getSampleCount(segmentElementsB);

	if (i < segmentSamplesA)
	{
		uint dstPos = binarySearchExclusive<1U>(d_Ranks[i], d_Ranks + segmentSamplesA, segmentSamplesB, nextPowerOfTwo(segmentSamplesB)) + i;
		d_Limits[dstPos] = d_Ranks[i];
	}

	if (i < segmentSamplesB)
	{
		uint dstPos = binarySearchInclusive<1U>(d_Ranks[segmentSamplesA + i], d_Ranks, segmentSamplesA, nextPowerOfTwo(segmentSamplesA)) + i;
		d_Limits[dstPos] = d_Ranks[segmentSamplesA + i];
	}
}

void Gpu::mergeRanksAndIndices(
		uint *d_LimitsA,
		uint *d_LimitsB,
		uint *d_RanksA,
		uint *d_RanksB,
		uint stride,
		uint N
)
{
	uint lastSegmentElements = N % (2 * stride);
	uint         threadCount = (lastSegmentElements > stride) ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE) : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

#pragma omp critical (launchLock)
	{
		setDevice();
		mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256, 0, stream>>>(
				d_LimitsA,
				d_RanksA,
				stride,
				N,
				threadCount
		);
		getLastCudaError("mergeRanksAndIndicesKernel(A)<<<>>> failed\n");
	}

#pragma omp critical (launchLock)
	{
		setDevice();
		mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256, 0, stream>>>(
				d_LimitsB,
				d_RanksB,
				stride,
				N,
				threadCount
		);
		getLastCudaError("mergeRanksAndIndicesKernel(B)<<<>>> failed\n");
	}
}



////////////////////////////////////////////////////////////////////////////////
// Merge step 3: merge elementary interRefs
////////////////////////////////////////////////////////////////////////////////
template<uint sortDir> inline __device__ void merge(
		KdCoord  coords[],
		KdCoord  *dstVal,
		refIdx_t *dstRef,
		KdCoord  *srcAVal,
		refIdx_t *srcARef,
		KdCoord  *srcBVal,
		refIdx_t *srcBRef,
		uint     lenA,
		uint     nPowTwoLenA,
		uint     lenB,
		uint     nPowTwoLenB,
		sint      p,
		sint      dim
)
{
	KdCoord  ValA, ValB;
	refIdx_t RefA, RefB;
	uint     dstPosA, dstPosB;

	if (threadIdx.x < lenA)
	{
		ValA = srcAVal[threadIdx.x];
		RefA = srcARef[threadIdx.x];
		dstPosA = binarySearchValExclusive<sortDir>(coords, ValA, RefA, srcBVal, srcBRef, lenB, nPowTwoLenB, p, dim) + threadIdx.x;
	}

	if (threadIdx.x < lenB)
	{
		ValB = srcBVal[threadIdx.x];
		RefB = srcBRef[threadIdx.x];
		dstPosB = binarySearchValInclusive<sortDir>(coords, ValB, RefB, srcAVal, srcARef, lenA, nPowTwoLenA, p, dim) + threadIdx.x;
	}

	__syncthreads();

	if (threadIdx.x < lenA)
	{
		dstVal[dstPosA] = ValA;
		dstRef[dstPosA] = RefA;
	}

	if (threadIdx.x < lenB)
	{
		dstVal[dstPosB] = ValB;
		dstRef[dstPosB] = RefB;
	}
}

template<uint sortDir> __global__ void mergeElementaryInterRefsKernel(
		KdCoord  coords[],
		KdCoord  *d_DstVal,
		refIdx_t *d_DstRef,
		KdCoord  *d_SrcVal,
		refIdx_t *d_SrcRef,
		uint     *d_LimitsA,
		uint     *d_LimitsB,
		uint     stride,
		uint     N,
		sint      p,
		sint     dim
)
{
	__shared__ KdCoord  s_Val[2 * SAMPLE_STRIDE];
	__shared__ refIdx_t s_Ref[2 * SAMPLE_STRIDE];

	const uint   interRefI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1);
	const uint   segmentBase = (blockIdx.x - interRefI) * SAMPLE_STRIDE;
	d_SrcVal += segmentBase;
	d_SrcRef += segmentBase;
	d_DstVal += segmentBase;
	d_DstRef += segmentBase;

	//Set up threadblock-wide parameters
	__shared__ uint startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

	if (threadIdx.x == 0)
	{
		uint segmentElementsA = stride;
		uint segmentElementsB = umin(stride, N - segmentBase - stride);
		uint segmentSamplesA = getSampleCount(segmentElementsA);
		uint segmentSamplesB = getSampleCount(segmentElementsB);
		uint segmentSamples = segmentSamplesA + segmentSamplesB;

		startSrcA    = d_LimitsA[blockIdx.x];
		startSrcB    = d_LimitsB[blockIdx.x];
		uint endSrcA = (interRefI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1] : segmentElementsA;
		uint endSrcB = (interRefI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1] : segmentElementsB;
		lenSrcA      = endSrcA - startSrcA;
		lenSrcB      = endSrcB - startSrcB;
		startDstA    = startSrcA + startSrcB;
		startDstB    = startDstA + lenSrcA;
	}

	//Load main input data
	__syncthreads();

	if (threadIdx.x < lenSrcA)
	{
		s_Val[threadIdx.x +             0] = d_SrcVal[0 + startSrcA + threadIdx.x];
		s_Ref[threadIdx.x +             0] = d_SrcRef[0 + startSrcA + threadIdx.x];
	}


	if (threadIdx.x < lenSrcB)
	{
		s_Val[threadIdx.x + SAMPLE_STRIDE] = d_SrcVal[stride + startSrcB + threadIdx.x];
		s_Ref[threadIdx.x + SAMPLE_STRIDE] = d_SrcRef[stride + startSrcB + threadIdx.x];
	}

	//Merge data in shared memory
	__syncthreads();
	merge<sortDir>(
			coords,
			s_Val,
			s_Ref,
			s_Val + 0,
			s_Ref + 0,
			s_Val + SAMPLE_STRIDE,
			s_Ref + SAMPLE_STRIDE,
			lenSrcA, SAMPLE_STRIDE,
			lenSrcB, SAMPLE_STRIDE,
			p, dim
	);


	//Store merged data
	__syncthreads();

	if (threadIdx.x < lenSrcA)
	{
		d_DstVal[startDstA + threadIdx.x] = s_Val[threadIdx.x];
		d_DstRef[startDstA + threadIdx.x] = s_Ref[threadIdx.x];
	}

	if (threadIdx.x < lenSrcB)
	{
		d_DstVal[startDstB + threadIdx.x] = s_Val[lenSrcA + threadIdx.x];
		d_DstRef[startDstB + threadIdx.x] = s_Ref[lenSrcA + threadIdx.x];
	}
}

void Gpu::mergeElementaryInterRefs(
		KdCoord  d_coords[],
		KdCoord  *d_DstVal,
		refIdx_t *d_DstRef,
		KdCoord  *d_SrcVal,
		refIdx_t *d_SrcRef,
		uint     *d_LimitsA,
		uint     *d_LimitsB,
		uint     stride,
		uint     N,
		uint     sortDir,
		sint     p,
		sint     dim
)
{
	uint lastSegmentElements = N % (2 * stride);
	uint          mergePairs = (lastSegmentElements > stride) ? getSampleCount(N) : (N - lastSegmentElements) / SAMPLE_STRIDE;

#pragma omp critical (launchLock)
	{
		setDevice();
		if (sortDir)
		{
			mergeElementaryInterRefsKernel<1U><<<mergePairs, SAMPLE_STRIDE, 0, stream>>>(
					d_coords,
					d_DstVal,
					d_DstRef,
					d_SrcVal,
					d_SrcRef,
					d_LimitsA,
					d_LimitsB,
					stride,
					N, p, dim
			);
			getLastCudaError("mergeElementaryInterRefsKernel<1> failed\n");
		}
		else
		{
			mergeElementaryInterRefsKernel<0U><<<mergePairs, SAMPLE_STRIDE, 0, stream>>>(
					d_coords,
					d_DstVal,
					d_DstRef,
					d_SrcVal,
					d_SrcRef,
					d_LimitsA,
					d_LimitsB,
					stride,
					N, p, dim
			);
			getLastCudaError("mergeElementaryInterRefsKernel<0> failed\n");
		}
	}
}

// uint *d_RanksA, *d_RanksB, *d_LimitsA, *d_LimitsB;
// uint maxSampleCount;
// sint *d_mpi;  //This is where the per partition merge path data will get stored.


void Gpu::initMergeSortSmpl(uint N)
{
#pragma omp critical (launchLock)
	{
		setDevice();
		maxSampleCount = 1 + (N / SAMPLE_STRIDE);
		checkCudaErrors(cudaMalloc((void **)&d_RanksA,  maxSampleCount * sizeof(uint)));
		checkCudaErrors(cudaMalloc((void **)&d_RanksB,  maxSampleCount * sizeof(uint)));
		checkCudaErrors(cudaMalloc((void **)&d_LimitsA, maxSampleCount * sizeof(uint)));
		checkCudaErrors(cudaMalloc((void **)&d_LimitsB, maxSampleCount * sizeof(uint)));
		checkCudaErrors(cudaMalloc((void **)&d_iRef, N*sizeof(refIdx_t)));
		checkCudaErrors(cudaMalloc((void **)&d_iVal, N*sizeof(KdCoord)));
	}
}

void Gpu::closeMergeSortSmpl()
{
#pragma omp critical (launchLock)
	{
		setDevice();
		syncGPU();
		checkCudaErrors(cudaFree(d_RanksA));
		checkCudaErrors(cudaFree(d_RanksB));
		checkCudaErrors(cudaFree(d_LimitsA));
		checkCudaErrors(cudaFree(d_LimitsB));
		checkCudaErrors(cudaFree(d_iRef));
		checkCudaErrors(cudaFree(d_iVal));
		if (d_mpi != NULL) checkCudaErrors(cudaFree(d_mpi));
	}
}

void Gpu::mergeSortSmpl(
		KdCoord  d_coords[],
		KdCoord  *d_DstVal,
		refIdx_t *d_DstRef,
		KdCoord  *d_BufVal,
		refIdx_t *d_BufRef,
		KdCoord  *d_SrcVal,
		refIdx_t *d_SrcRef,
		uint     N,
		uint     sortDir,
		sint      p,
		sint      dim
)
{
	uint stageCount = 0;

	for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1, stageCount++);

	refIdx_t *iRef, *oRef;
	KdCoord  *iVal, *oVal;

	if (stageCount & 1)
	{
		iVal = d_BufVal;
		iRef = d_BufRef;
		oVal = d_DstVal;
		oRef = d_DstRef;
	}
	else
	{
		iVal = d_DstVal;
		iRef = d_DstRef;
		oVal = d_BufVal;
		oRef = d_BufRef;
	}

	assert(N <= (SAMPLE_STRIDE * maxSampleCount));
	assert(N % SHARED_SIZE_LIMIT == 0);

	//#define PRINT_TIME
#ifdef PRINT_TIME
float time;
	cudaEvent_t t_start, t_stop;
	checkCudaErrors(cudaEventCreate(&t_start));
	checkCudaErrors(cudaEventCreate(&t_stop));

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventRecord(t_start));
#endif

	mergeSortShared(d_coords, iVal, iRef, d_SrcVal, d_SrcRef, N / SHARED_SIZE_LIMIT, SHARED_SIZE_LIMIT, sortDir, p, dim);

	for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1)
	{
		uint lastSegmentElements = N % (2 * stride);

		//Find sample ranks and prepare for limiters merge
		generateSampleRanks(d_coords, d_RanksA, d_RanksB, iVal, iRef, stride, N, sortDir, p, dim);

		//Merge ranks and indices
		mergeRanksAndIndices(d_LimitsA, d_LimitsB, d_RanksA, d_RanksB, stride, N);

		//Merge elementary interRefs
		mergeElementaryInterRefs(d_coords, oVal, oRef, iVal, iRef, d_LimitsA, d_LimitsB, stride, N, sortDir, p, dim);

		if (lastSegmentElements <= stride)
		{
			//Last merge segment consists of a single array which just needs to be passed through
			checkCudaErrors(cudaMemcpyAsync(oVal + (N - lastSegmentElements), iVal + (N - lastSegmentElements), lastSegmentElements * sizeof(KdCoord), cudaMemcpyDeviceToDevice, stream));
			checkCudaErrors(cudaMemcpyAsync(oRef + (N - lastSegmentElements), iRef + (N - lastSegmentElements), lastSegmentElements * sizeof(uint)  , cudaMemcpyDeviceToDevice, stream));
		}

		KdCoord* tv = iVal;
		iVal = oVal;
		oVal = tv;
		refIdx_t* tr = iRef;
		iRef = oRef;
		oRef = tr;
	}
#ifdef PRINT_TIME
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventRecord(t_stop));
	checkCudaErrors(cudaEventSynchronize(t_stop));
	checkCudaErrors(cudaEventElapsedTime(&time, t_start, t_stop));
	printf ("Sort took %f seconds\n",time/1000.0);
	checkCudaErrors(cudaEventDestroy(t_start));
	checkCudaErrors(cudaEventDestroy(t_stop));
#endif

}


/*  Multi GPU sort algorithm is an implementation of the algorithm presented in
Comparison Based Sorting for Systems with MultipleGPUs
by Ivan Tanasic, Lluís Vilanova, Marc Jord , Javier Cabezas, Isaac Gelado,  Nacho Navarro, and  Wen-mei Hwu
 */

// __device__ uint d_pivot;

template<uint sortDir> __global__ void pivotSelection(KdCoord coordA[], KdCoord valA[], refIdx_t refA[],
		KdCoord coordB[], KdCoord valB[], refIdx_t refB[],
		sint p, sint dim, uint L,
		uint* d_pivot)
{
	if (L == 0)  {
		*d_pivot = 0;
		return;
	}

	uint pivot = L/2;
	uint stride = pivot/2;
	for (; stride > 0; stride >>= 1) {
		//	if a[len(a) − pivot − 1] < b[pivot] then
		if (( sortDir && (superKeyCompareFirstDimSmpl(valA[L - pivot - 1], valB[pivot], coordA+refA[L- pivot - 1]*dim, coordB+refB[pivot]*dim, p, dim) < 0)) ||
				(!sortDir && (superKeyCompareFirstDimSmpl(valA[L - pivot - 1], valB[pivot], coordA+refA[L- pivot - 1]*dim, coordB+refB[pivot]*dim, p, dim) > 0))) {
			//  if a[len(a) − pivot − 2] < b[pivot + 1] &
			if ((( sortDir && (superKeyCompareFirstDimSmpl(valA[L - pivot - 2], valB[pivot + 1], coordA+refA[L- pivot - 2]*dim, coordB+refB[pivot + 1]*dim, p, dim) < 0)) ||
					(!sortDir && (superKeyCompareFirstDimSmpl(valA[L - pivot - 2], valB[pivot + 1], coordA+refA[L- pivot - 2]*dim, coordB+refB[pivot + 1]*dim, p, dim) > 0))) &&
					//  a[len(a) − pivot] > b[pivot − 1] then
					(( sortDir && (superKeyCompareFirstDimSmpl(valA[L - pivot], valB[pivot - 1], coordA+refA[L- pivot]*dim, coordB+refB[pivot - 1]*dim, p, dim) > 0)) ||
							(!sortDir && (superKeyCompareFirstDimSmpl(valA[L - pivot], valB[pivot - 1], coordA+refA[L- pivot]*dim, coordB+refB[pivot - 1]*dim, p, dim) < 0)))) {
				*d_pivot = pivot;
				return;
			} else {
				pivot = pivot - stride;
			}
		} else {
			pivot = pivot + stride;
		}
	} // end for
	if (pivot == 1 && (( sortDir && (superKeyCompareFirstDimSmpl(valA[L - 1], valB[0], coordA+refA[L - 1]*dim, coordB+refB[0]*dim, p, dim) < 0)) ||
			(!sortDir && (superKeyCompareFirstDimSmpl(valA[L - 1], valB[0], coordA+refA[L - 1]*dim, coordB+refB[0]*dim, p, dim) > 0)))) {
		pivot = 0;
	} else if (pivot == (L-1) && (( sortDir && (superKeyCompareFirstDimSmpl(valA[0], valB[L - 1], coordA+refA[0]*dim, coordB+refB[L - 1]*dim, p, dim) > 0)) ||
			(!sortDir && (superKeyCompareFirstDimSmpl(valA[0], valB[L - 1], coordA+refA[0]*dim, coordB+refB[L - 1]*dim, p, dim) < 0)))) {
		pivot = L;
	}
	*d_pivot = pivot;
}

static __global__ void pivotSwap(KdCoord coordA[], KdCoord valA[], refIdx_t refA[],
		KdCoord coordB[], KdCoord valB[], refIdx_t refB[],
		sint p, sint dim, uint L, uint* d_pivot){
	uint numThreads = gridDim.x * blockDim.x;
	uint pos = blockIdx.x * blockDim.x + threadIdx.x;
	uint pivot = *d_pivot;
	refA   += (L - pivot);
	valA   += (L - pivot);
	for (uint swapi = pos; swapi < (pivot)*dim; swapi += numThreads) {
		uint swapP = swapi / dim;
		uint swap =  swapi - swapP * dim;
		refIdx_t Aref = refA[swapP]*dim+swap;
		refIdx_t Bref = refB[swapP]*dim+swap;

		KdCoord A = coordA[Aref];
		KdCoord B = coordB[Bref];
		coordA[Aref] = B;
		coordB[Bref] = A;
		if (swap == p){
			valA[swapP] = B;
			valB[swapP] = A;
		}
	}

}


uint Gpu::balancedSwap(KdCoord* coordA, KdCoord* valA, refIdx_t* refA,
		KdCoord* coordB, KdCoord *valB, refIdx_t* refB,
		sint sortDir, sint p, sint dim, uint NperG, sint numThreads){

	uint pivot;


#pragma omp critical (launchLock)
	{
		setDevice();
		if (sortDir == 0) {
			pivotSelection<0U><<<1,1, 0, stream>>>(coordA, valA, refA, coordB, valB, refB, p, dim, NperG, d_pivot);
		} else {
			pivotSelection<1U><<<1,1, 0, stream>>>(coordA, valA, refA, coordB, valB, refB, p, dim, NperG, d_pivot);
		}
		checkCudaErrors(cudaGetLastError());
	}
	// cudaMemcpyFromSymbolAsync(&pivot, d_pivot, sizeof(pivot), 0,cudaMemcpyDeviceToHost, stream);
	checkCudaErrors(cudaMemcpyAsync(&pivot, d_pivot, sizeof(uint), cudaMemcpyDeviceToHost, stream));

#pragma omp critical (launchLock)
	{
		setDevice();
		pivotSwap<<<numThreads,1024, 0, stream>>>(coordA, valA, refA, coordB, valB, refB, p, dim, NperG, d_pivot);
		checkCudaErrors(cudaGetLastError());
	}
	return pivot;
}

#define min(a,b) ((a)<(b) ? (a) : (b))
#define max(a,b) ((a)>(b) ? (a) : (b))

template <uint sortDir>
__device__ uint mergePath(KdCoord coord[], KdCoord valA[], refIdx_t refA[], uint aCount,
		KdCoord valB[], refIdx_t refB[], int bCount,
		sint diag, sint p, sint dim,  uint N) {

	sint begin = max(0, diag - bCount);
	sint end = min(diag, aCount);

	while(begin < end) {
		uint mid = begin + ((end - begin) >> 1);
		KdCoord  aVal = valA[mid];
		KdCoord  bVal = valB[diag - 1 - mid];
		refIdx_t aRef = refA[mid];
		refIdx_t bRef = refB[diag - 1 - mid];
		bool pred = ( sortDir && (superKeyCompareFirstDimSmpl(aVal, bVal, coord+aRef*dim, coord+bRef*dim, p, dim) < 0)) ||
				(!sortDir && (superKeyCompareFirstDimSmpl(aVal, bVal, coord+aRef*dim, coord+bRef*dim, p, dim) > 0));
		if(pred) begin = mid + 1;
		else end = mid;
	}
	return begin;
}

template <uint sortDir>
__global__ void getMergePaths(KdCoord coord[], KdCoord valA[], refIdx_t refA[], uint aCount,
		KdCoord valB[], refIdx_t refB[], int bCount,
		sint mpi[], sint p, sint dim,  uint N){

	uint numThreads = gridDim.x * blockDim.x;
	uint pos = blockIdx.x * blockDim.x + threadIdx.x;

	uint partitions = iDivUp((aCount + bCount),MERGE_PATH_BLOCK_SIZE);
	if (pos == 0) {
		mpi[0] = 0;  mpi[partitions] = aCount;
	}
	for (uint i = pos + 1; i < partitions; i += numThreads){
		uint diag = i * MERGE_PATH_BLOCK_SIZE;
		mpi[i] = mergePath<sortDir>(coord, valA, refA, aCount, valB, refB, bCount,
				diag, p, dim, N);
	}
}


/*
 * mergePartitions
 *
 * The mergePartitions Function merges the the data from two value and reference arrays into 1 in sorterd order.
 * The data in the two arrays must bw sorted.
 *
 *  KdCoord coord[]		Coordinates array
 *  KdCoord valASrc[]	First Array of the primary coordinate to be sorted on in sorted order
 *  refIdx_t refASrc[]  First Array of the references into the coordinate array corresponding to the value array
 *  uint aCount			length of first array
 *  KdCoord valASrc[]	Second Array of the primary coordinate to be sorted on in sorted order
 *  refIdx_t refASrc[]  Second Array of the references into the coordinate array corresponding to the value array
 *  uint bCount			length of second array
 *  KdCoord valDst[]	Array to place the resulting merged values
 *  refIdx_t refDst[]	Array to place the resulting merged references
 *  sint mpi[]			Array of partition indices indicating merge points at all of the partitions
 *  sint p				index of the primary coordinate
 *  sint dim			number of coordinates in a tuple
 *  uint N				total length of the resulting array
 */
template < uint sortDir >
__global__ void mergePartitions(KdCoord coord[], KdCoord valASrc[], refIdx_t refASrc[], uint aCount,
		KdCoord valBSrc[], refIdx_t refBSrc[], int bCount, KdCoord valDst[], refIdx_t refDst[],
		sint mpi[], sint p, sint dim,  uint N) {

	// Allocate shared memory
	__shared__ KdCoord  sharedValIn[MERGE_PATH_BLOCK_SIZE];
	__shared__ refIdx_t sharedRefIn[MERGE_PATH_BLOCK_SIZE];
	__shared__ KdCoord  sharedValOut[MERGE_PATH_BLOCK_SIZE];
	__shared__ refIdx_t sharedRefOut[MERGE_PATH_BLOCK_SIZE];

	KdCoord  val;
	refIdx_t ref;

	uint partitions = iDivUp((N),MERGE_PATH_BLOCK_SIZE);  // Recalculate the number of partitions to process
	for (uint bid = blockIdx.x;  bid < partitions; bid+=gridDim.x){  // And loop through those partitions
		uint grid = bid * MERGE_PATH_BLOCK_SIZE;		// Calculate the relevant index ranges into the two source arrays
		uint a0 = mpi[bid];								// for this partition
		uint a1 = mpi[bid+1];
		uint b0 = grid - mpi[bid];
		uint b1 = min(aCount + bCount, grid + MERGE_PATH_BLOCK_SIZE) - mpi[bid+1];
		sint wtid = threadIdx.x + bid * MERGE_PATH_BLOCK_SIZE;  //Place there this thread will write the data

		if (a0 == a1) {							// If no a data just copy b
			val = valBSrc[b0+threadIdx.x];
			ref = refBSrc[b0+threadIdx.x];
			valDst[wtid] = val;
			refDst[wtid] = ref;
		} else if (b0 == b1) {					// If no b data just copy a
			val = valASrc[a0+threadIdx.x];
			ref = refASrc[a0+threadIdx.x];
			valDst[wtid] = val;
			refDst[wtid] = ref;
		} else {
			bool abs = threadIdx.x < a1-a0;					//Is this tread working on a or b data?
			if (abs) {							// Read in data to be merged and store in shared memory
				uint rtid = a0 + threadIdx.x;
				val = valASrc[rtid];
				ref = refASrc[rtid];
			} else {
				uint rtid = b0 + threadIdx.x - (a1 - a0);
				val = valBSrc[rtid];
				ref = refBSrc[rtid];
			}
			sharedValIn[threadIdx.x] = val;			// Copy the data to shared memory
			sharedRefIn[threadIdx.x] = ref;
			__syncthreads();					// Wait for all threads in the block to be done
			// Now sort the merge the data into the output buffer using the binary search method
			sint xns  = abs ? b1 - b0 : a1 - a0; // Size of the other array to search
			sint soff = abs ? a1 - a0 : 0;		// Offset into the input array of the other buffer
			sint woff = abs ? 0		  : a1 - a0; // Offset from TID of the write bugger.
			uint xo = binarySearchVal<sortDir>(abs, coord, val, ref, sharedValIn + soff, sharedRefIn + soff, xns, nextPowerOfTwo(xns), p, dim);
			sharedValOut[xo + threadIdx.x - woff] = val;
			sharedRefOut[xo + threadIdx.x - woff] = ref;

			__syncthreads();					// And wait for all threads in the block to be done
			valDst[wtid] = sharedValOut[threadIdx.x];   // Write the merged data to global memory
			refDst[wtid] = sharedRefOut[threadIdx.x];
		}
		// Since the next operation in the loop is to read into the input buffer, no collision so no need to synchronize a third time.
	}
}

/*
 * mergeSwap
 * The merge swap function handles the second half of the dual GPU sorting function.  After the sorted
 * data has been swapped between the two GPUs, this function merges the swapped data with the kept data.
 * The two sets of data is assumed to be in the same source array with the dividing line at the merge
 * point.  This function needs to be called twice, once for each GPU
 *  KdCoord d_coord[]		Coordinates array
 *  KdCoord d_valASrc[]	Source Array of the primary coordinate to be sorted on in sorted order
 *  refIdx_t d_refASrc[]  Source Array of the references into the coordinate array corresponding to the value array
 *  KdCoord d_valDst[]	Array to place the resulting merged values
 *  refIdx_t d_refDst[]	Array to place the resulting merged references
 *  sint mergePnt		dividing point in the arrays between the data to be merged
 *  sint p				index of the primary coordinate
 *  sint dim			number of coordinates in a tuple
 *  uint N				total length of the resulting array
 */
void Gpu::mergeSwap(KdCoord d_coord[], KdCoord d_valSrc[], refIdx_t d_refSrc[],
		KdCoord d_valDst[], refIdx_t d_refDst[],
		sint mergePnt, sint p, sint dim,  uint N, sint numThreads){

	uint partitions = iDivUp(N,MERGE_PATH_BLOCK_SIZE);
	uint aCount = mergePnt;
	uint bCount = N - mergePnt;

#pragma omp critical (launchLock)
	{
		setDevice();
		checkCudaErrors(cudaMalloc((void **)&d_mpi,  (partitions+1) * sizeof(sint)));
		// Call getMergePaths to find the merge bounds of each block
		getMergePaths<1u><<<iDivUp(numThreads,MERGE_PATH_BLOCK_SIZE),MERGE_PATH_BLOCK_SIZE, 0, stream>>>(d_coord, d_valSrc, d_refSrc, aCount,
				d_valSrc + aCount, d_refSrc + aCount, bCount,
				d_mpi, p, dim, N);
		checkCudaErrors(cudaGetLastError());
	}

#pragma omp critical (launchLock)
	{
		setDevice();
		// Call the mergePartitions kernel to merge each block.
		mergePartitions<1U><<<iDivUp(numThreads,MERGE_PATH_BLOCK_SIZE),MERGE_PATH_BLOCK_SIZE, 0, stream>>>(d_coord, d_valSrc, d_refSrc, aCount,
				d_valSrc + aCount, d_refSrc + aCount, bCount,
				d_valDst, d_refDst, d_mpi, p, dim,  N);
		checkCudaErrors(cudaGetLastError());
	}

}


