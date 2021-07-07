//
//  buildTree.cu
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
 * The partitioning algorithm uses an approach based on the following:
 * "Efficient Stream Compaction on Wide SIMD Many-Core Architectures"
 * by Markus Billeter, Ola Olsson, Ulf Assarsson
 * http://www.cse.chalmers.se/~uffe/streamcompaction.pdf
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
using std::setprecision;
using namespace std;
#include <assert.h>
#include <helper_cuda.h>
#include <sm_30_intrinsics.h>

#include "buildKdTree_common.h"
#include "Gpu.h"


__device__ KdCoord superKeyCompareB(const KdCoord *a, const KdCoord *b, const sint p, const sint dim)
{
	KdCoord diff = a[p] - b[p];
	for (sint i = 1; diff == 0 && i < dim; i++) {
		sint r = i + p;
		r = (r < dim) ? r : r - dim;
		diff = a[r] - b[r];
	}
	return diff;
}

__device__ KdCoord superKeyComparePD(const KdCoord ap, const KdCoord bp, const KdCoord *a, const KdCoord *b, const sint p, const sint dim)
{
	KdCoord diff = ap - bp;
	for (sint i = 1; diff == 0 && i < dim; i++) {
		sint r = i + p;
		r = (r < dim) ? r : r - dim;
		diff = a[r] - b[r];
	}
	return diff;
}


/*
 * Check the validity of the merge sort and remove duplicates from a reference array.
 *
 * calling parameters:
 *
 * reference - a vector<int*> that represents one of the reference arrays
 * i - the leading dimension for the super key
 * dim - the number of dimensions
 *
 * returns: the end index of the reference array following removal of duplicate elements
 */
__device__ void cuWarpCopyRef(refIdx_t refout[], refIdx_t refin[], sint segSize, const sint numTuples) {

	uint pos = blockIdx.x * blockDim.x + threadIdx.x;
	uint thrdIdx = (pos & (warpSize-1));
	uint warpsPerBlock = (SHARED_SIZE_LIMIT/(2*warpSize));
	uint warpIndex = ((pos - thrdIdx)/warpSize);
	refIdx_t ref;

	if (segSize < warpSize*200) {  //The copy is small, so do a simple unaligned copy and return
		for (sint j = 0;  j+thrdIdx < segSize; j += warpSize){
			refout[j+thrdIdx] = refin[j+thrdIdx];
		}
		return;
	}

	// allocate the shared memory that will be used for coalescing of writes.
	__shared__ refIdx_t s_ref[SHARED_SIZE_LIMIT];
	__shared__ uint     s_tag[SHARED_SIZE_LIMIT];
	// allocate the input and output counter
	uint   outCnt, oldOutCnt;
	uint   inCnt;

	// Calculate the base index for this warp in the shared memory array
	// SHARED_SIZE_LIMIT/(2*warpSize) is the number of warps per block
	// so the warp in block index is the mod of warpIndex by the num warps in block.
	uint sharedBase = 2 * warpSize *  (warpIndex % warpsPerBlock);
	uint sharedAddrMask = (2*warpSize)-1;

	// clear the dirty tags
	s_tag[sharedBase + thrdIdx] 		 = 0;
	s_tag[sharedBase + warpSize + thrdIdx] = 0;

	// come up with warpSize word aligned base write address
	// first calculate the warp aligned read address below the starting address
	refIdx_t*   refptr = (refIdx_t*)((ulong)refout & ~((warpSize*sizeof(refIdx_t)) -1));
	// initialize the output counter to be relative to the warpSize aligned write buffers
	outCnt = int(refout - refptr);
	refout = refptr;

	// Do the first reads to align the input pointers to warpSize word boundary
	// First calculate the warp aligned read address below the starting address
	refptr = (refIdx_t*)  ((ulong)refin & ~((warpSize*sizeof(refIdx_t)) -1));
	// Calculate the input counter
	inCnt = warpSize + refptr - refin;
	// then read the words from the input only up to the next warpSize Boundary
	// and write to shared memory as indexed by the output counter
	if (thrdIdx < inCnt) {
		ref = refin[thrdIdx];
		s_ref[sharedBase + ((outCnt + thrdIdx) & sharedAddrMask)] = ref;
		s_tag[sharedBase + ((outCnt + thrdIdx) & sharedAddrMask)] = 1;
	}
	// Increment the aligned input pointer
	refin = refptr + warpSize;
	// Update the output counters
	oldOutCnt = outCnt;
	outCnt += inCnt;

	// If the last read crossed the boundary of the coalescing buffers, write out the valid words in the old buffer
	if (((oldOutCnt ^ outCnt) & warpSize) != 0) {
		if (s_tag[sharedBase + (oldOutCnt & warpSize) + thrdIdx] == 1) {
			refout[(oldOutCnt & ~(warpSize-1)) + thrdIdx] = s_ref[sharedBase + (oldOutCnt & warpSize) + thrdIdx];
			s_tag[sharedBase + (oldOutCnt & warpSize) + thrdIdx] = 0;
		}
	} else { // Else read another warp's worth to prime the buffer
		ref = refin[thrdIdx];
		s_ref[sharedBase + ((outCnt + thrdIdx) & sharedAddrMask)] = ref;
		s_tag[sharedBase + ((outCnt + thrdIdx) & sharedAddrMask)] = 1;
		oldOutCnt = outCnt;
		outCnt += warpSize;
		if (((oldOutCnt ^ outCnt) & warpSize) != 0) {
			if (s_tag[sharedBase + (oldOutCnt & warpSize) + thrdIdx] == 1) {
				refout[(oldOutCnt & ~(warpSize-1)) + thrdIdx] = s_ref[sharedBase + (oldOutCnt & warpSize) + thrdIdx];
				s_tag[sharedBase + (oldOutCnt & warpSize) + thrdIdx] = 0;
			}
		}
		// Increment the input counter
		inCnt += warpSize;
		// Increment the aligned input pointer
		refin += warpSize;
	}

	// OK, input pointer is now at a warSize addr boundary and the coalesce buffer has been primed.
	// Time to go into the main loop  The loop will count through the remaining inputs

	while (inCnt < segSize) {
		if (inCnt+thrdIdx < segSize) {
			ref = refin[thrdIdx];
			s_ref[sharedBase + ((outCnt + thrdIdx) & sharedAddrMask)] = ref;
			s_tag[sharedBase + ((outCnt + thrdIdx) & sharedAddrMask)] = 1;
		}
		oldOutCnt = outCnt;
		outCnt += inCnt+warpSize <= segSize ? warpSize : segSize - inCnt;

		if (((oldOutCnt ^ outCnt) & warpSize) != 0) {
			if (s_tag[sharedBase + (oldOutCnt & warpSize) + thrdIdx] == 1) {
				refout[(oldOutCnt & ~((warpSize)-1)) + thrdIdx] = s_ref[sharedBase + (oldOutCnt & warpSize) + thrdIdx];
				s_tag[sharedBase + (oldOutCnt & warpSize) + thrdIdx] = 0;
			}
		}
		// Increment the input counter
		inCnt += warpSize;
		// Increment the aligned input pointer
		refin += warpSize;
	}
	// Write out the final buffer
	if (s_tag[sharedBase + (outCnt & warpSize) + thrdIdx] == 1) {
		refout[(outCnt & ~(warpSize-1)) + thrdIdx] = s_ref[sharedBase + (outCnt & warpSize) + thrdIdx];
		s_tag[sharedBase + (outCnt & warpSize) + thrdIdx] = 0;
	}

}

__global__ void cuPartitionRemoveGaps(refIdx_t refoutx[], refIdx_t refinxLT[], refIdx_t refinxGT[], uint segLengthsLT[],
		uint segLengthsGT[], const sint startIn, const sint endIn, sint level) {

	uint pos = (blockIdx.x * blockDim.x + threadIdx.x);
	uint allWarps = gridDim.x * blockDim.x / warpSize;
	uint thrdIdx = (pos & (warpSize-1));
	uint warpIndex = ((pos - thrdIdx)/warpSize);

	uint start = startIn;
	uint end = endIn;
	uint mid;

	for (uint i = 0;  i < level; i++) {
		mid = start + ((end - start)>>1);
		if (warpIndex & (allWarps >> (i+1))) {
			start = mid + 1;
		} else {
			end = mid -1;
		}
	}
	mid = start + ((end - start)>>1);
	sint partSize = end-start+1;
	uint segSize = (partSize + (allWarps>>level) - 1) / (allWarps>>level);
	uint segStart = start + segSize * (warpIndex - (warpIndex & ~((allWarps >> level) - 1)));
	// do the simple slow implementation first
	// get the seg start and seg size from the segLentghs array written by the partition functions
	// sum up the lengths of all of the lengths of the segments at a lower index than this segment
	// start at the base of the warp group.  Do the LT data copy first
	uint segStartOut = start;
	if (thrdIdx == 0) {
		for (uint i = (warpIndex & ~((allWarps >> level) - 1)); i < warpIndex; i++)
			segStartOut += segLengthsLT[i];
		segSize = segLengthsLT[warpIndex];
	}
	// Copy to the other threads in the warp.
	segStartOut = __shfl(segStartOut, 0);
	segSize = __shfl(segSize, 0);
	// and do the copy.
	cuWarpCopyRef(refoutx+segStartOut, refinxLT+segStart, segSize, partSize);
	// Check to see that the partitioned data did not exceed it's half of the output array.
	sint partitionCount = segStartOut + segLengthsLT[warpIndex];
	if (partitionCount > (mid)) {
		return; //TODO should add an assert here;
	}

	// do the copy again for the gt data
	segStartOut = mid+1;
	if (thrdIdx == 0) {
		for (uint i = (warpIndex & ~((allWarps >> level) - 1)); i < warpIndex; i++)
			segStartOut += segLengthsGT[i];
		segSize = segLengthsGT[warpIndex];
	}
	// Copy to the other threads in the warp.
	segStartOut = __shfl(segStartOut, 0);
	segSize = __shfl(segSize, 0);
	// and do the copy.+
	cuWarpCopyRef(refoutx+segStartOut, refinxGT+segStart, segSize, partSize);

	// Check to see that the partitioned data did not exceed it's half of the output array.
	partitionCount = segStartOut + segLengthsGT[warpIndex];
	if (partitionCount > (end+1)) {
		return; //TODO should add an assert here;
	}
}

#define SIMPLE_COPY
#ifdef SIMPLE_COPY

__global__ void cuCopyRef(refIdx_t refout[], refIdx_t refin[], const sint numRefs) {

	uint allThreads = gridDim.x * blockDim.x;  // Total number of warps started
	uint pos = blockIdx.x * blockDim.x + threadIdx.x;
	for (sint i = pos; i<numRefs; i += allThreads)
		refout[i] = refin[i];
}

#else
__global__ void cuCopyRef(refIdx_t refoutx[], refIdx_t refinx[], const sint numTuples) {

	uint allWarps = gridDim.x * blockDim.x / warpSize;  // Total number of warps started
	uint pos = blockIdx.x * blockDim.x + threadIdx.x;
	uint thrdIdx = (pos & (warpSize-1));
	//    uint warpsPerBlock = (SHARED_SIZE_LIMIT/(2*warpSize));
	uint warpIndex = ((pos - thrdIdx)/warpSize);
	uint segSize = (numTuples + allWarps - 1) / (allWarps);

	// calculate the base addrs of the global memory input and output arrays.
	uint segStart = warpIndex * segSize;
	if (segStart + segSize > numTuples) {
		segSize = numTuples - segStart;
	}

	cuWarpCopyRef(refoutx + segStart,  refinx + segStart, segSize, numTuples);
}
#endif


__device__ sint d_partitionError;
#define PART_SIZE_GT_SUB_PART_SIZE -1
#define PART_FINISH_DELTA_TOO_LARGE -2

__device__ void cuSmallPartition( const __restrict__ KdCoord coords[],
		refIdx_t refoutxLT[], refIdx_t refoutxGT[], refIdx_t refinx[],
		const refIdx_t divRef, const sint p, const sint dim, uint segSizex,
		const uint subWarpSize)
{
	uint pos = (blockIdx.x * blockDim.x + threadIdx.x); // thread ID
	uint thrdIdx = (pos & (warpSize-1)); // Index within the warp
	uint subWarpIdx = thrdIdx / subWarpSize; // subWarp index within the warp
	uint subThrdIdx = thrdIdx - subWarpIdx * subWarpSize; // Thread index within the subWarp
	uint subWarpMask = ((1<<subWarpSize)-1) << subWarpIdx * subWarpSize; // subWarp Mask
	uint segSize;
	uint outCntLT = 0;
	uint outCntGT = 0;

	segSize = segSizex;

	refIdx_t* refin = refinx;
	refIdx_t* refoutLT = refoutxLT;
	refIdx_t* refoutGT = refoutxGT;

	KdCoord divVal = coords[divRef*dim+p];

	KdCoord cmp = 0;
	uint maskGEme = ((1 << thrdIdx) - 1);

	uint ref;
	if (subThrdIdx < segSize) {// inside the segment?
		ref = refin[subThrdIdx];
		// do the compare
		KdCoord val = coords[ref*dim+p];
		cmp = superKeyComparePD(val, divVal, coords+ref*dim, coords+divRef*dim, p, dim);
	} else {
		cmp = 0; // Use cmp == 0 to exclude data outside the segment
	}
	refin += warpSize;
	// Write out the less than indices
	uint shflMask = __ballot(cmp<0) & subWarpMask;
	if (cmp < 0) {
		// Calculate the address which is determined by the number of kept values less than this thread.
		sint wrtIdx = __popc(shflMask & maskGEme);
		refoutLT[(outCntLT + wrtIdx)] = ref;
	}
	// Update the output counter
	outCntLT += __popc(shflMask);
	// Write out the greater than values
	shflMask = __ballot(cmp>0) & subWarpMask;
	if (cmp > 0) {
		// Calculate the address which is determined by the number of kept values less than this thread.
		sint wrtIdx = __popc(shflMask & maskGEme);
		refoutGT[(outCntGT + wrtIdx)] = ref;
	}
	// Update the output counter
	outCntGT += __popc(shflMask);

}

__global__ void cuPartitionShort( KdNode kdNodes[], const __restrict__ KdCoord coords[],
		refIdx_t refoutx[], refIdx_t refinx[], refIdx_t refp[],
		const sint p, const sint dim,
		refIdx_t midRefs[], refIdx_t lastMidRefs[],
		sint startIn, sint endIn,
		const sint level, const sint logNumSubWarps, const sint logSubWarpSize)
{
	uint pos = (blockIdx.x * blockDim.x + threadIdx.x); // This thread's position in all threads
	uint subWarpSize = 1<<logSubWarpSize;
	uint allSubWarps = gridDim.x * blockDim.x / subWarpSize;  // Total number of subWarps started
	uint subThrdIdx = (pos & (subWarpSize-1));  // this threads position in the subWarp.
	uint subWarpIndex = (pos - subThrdIdx)/(subWarpSize); // this subWarps position in all threads
	uint loopLevels = level-logNumSubWarps;                // log of the nuber of iterations to be done

	// This first loop iterates over the partition regions when there are more portion regions then thread.
	// Note that if the there are more warps than partition regions (level < logNumWarps) the iteration count will be 0
	// and start and end will be untouched.
	for (uint loop = 0; loop < (1<<loopLevels); loop++) {
		uint start = startIn;                               // local copy of start
		uint end = endIn;                                   // local copy of end
		uint mid;                                           // mid between start and end

		// This loop determines the start and end of the current iteration over the partitions.
		for (uint k = 1;  k <= loopLevels; k++) {
			mid = start + ((end - start)>>1);
			if (loop & (1 << (loopLevels - k )))
			{ start = mid + 1; } else { end = mid -1; }
		}

		// Now calculate the start and end  end and mid using the iterative methode for this warps partition segment.
		for (uint i = 0;  i < (logNumSubWarps); i++) {
			mid = start + ((end - start)>>1);
			if (subWarpIndex & (allSubWarps >> (i+1)))
			{ start = mid + 1; } else { end = mid -1; }
		}

		if((end - start + 1) > subWarpSize) {
			d_partitionError =  PART_SIZE_GT_SUB_PART_SIZE;
		}

		mid = start + ((end - start)>>1);
		// Calculate the size of the partition segment that this warp will partition.
		sint partSize = end - start + 1; // number of reference to partition
		// get the reference to the coordinate that will be partitioned against.
		refIdx_t midRef = refp[mid];
		cuSmallPartition( coords, // pointer to coordinate array
				refoutx+start, // pointer to the beginning of the output ref array for this subdivision
				refoutx+mid+1, // pointer to the beginning of the output ref array for this subdivision
				refinx+start, // pointer to the beginning of the input ref array for this subdivision
				midRef, // reference to coordinate against which spitting the arrays will be partitioned
				p, dim, // which dimension is being used for partitioning and the number of dimensions
				partSize, // The size of the segment that each warp will partition
				subWarpSize // size of the subwarp
		);

		// if this thread is the 0th thread and this warp is starting warp in a warp group, save off the mid point.
		if (subThrdIdx == 0 ) {
			uint mra = subWarpIndex + loop * allSubWarps;
			midRefs[mra] = midRef;
			if (lastMidRefs != NULL) {
				if (mra & 1) { // odd or even?
					kdNodes[lastMidRefs[mra>>1]].gtChild = midRef;
				} else {
					kdNodes[lastMidRefs[mra>>1]].ltChild = midRef;
				}
			}
		}
	}
}

__device__ void cuSinglePartition( const __restrict__ KdCoord coords[], refIdx_t refoutxLT[], refIdx_t refoutxGT[], refIdx_t refinx[],
		const refIdx_t divRef, const sint p, const sint dim, uint segSizex, uint segLengthsLT[], uint segLengthsGT[],
		const sint numTuples, uint warpGroupSize)
{
	uint pos = (blockIdx.x * blockDim.x + threadIdx.x);
	uint thrdIdx = (pos & (warpSize-1));
	uint warpsPerBlock = (SHARED_SIZE_LIMIT/(2*warpSize));
	uint warpIndex = ((pos - thrdIdx)/warpSize) % warpGroupSize;
	uint segSize;
	uint outCntLT = 0;
	uint outCntGT = 0;
	uint oldOutCntLT;
	uint oldOutCntGT;
	refIdx_t ref;

	// Calculate the base addrs of the global memory input and output arrays.
	uint segStart = warpIndex * segSizex;
	if (segStart + segSizex > numTuples) {
		segSize = numTuples - segStart;
	} else segSize = segSizex;

	refIdx_t* refin = refinx + segStart;
	refIdx_t* refoutLT = refoutxLT + segStart;
	refIdx_t* refoutGT = refoutxGT + segStart;

	// Allocate the shared memory that will be used for coalescing of writes.
	__shared__ refIdx_t s_refLT[SHARED_SIZE_LIMIT];
	__shared__ refIdx_t s_refGT[SHARED_SIZE_LIMIT];

	KdCoord divVal = coords[divRef*dim+p];

	// Calculate the base index for this warp in the shared memory array
	// SHARED_SIZE_LIMIT/(2*warpSize) is the number of warps per block
	// so the warp in block index is the mod of warpIndex by the num warps in block.
	uint sharedBase = 2 * warpSize *  (((pos - thrdIdx)/warpSize) % warpsPerBlock);
	uint sharedAddrMask = (2*warpSize)-1;

	KdCoord cmp = 0;
	uint maskGEme = ((1 << thrdIdx) - 1);

	//  Now start looping
	for (sint j = 0;  j < segSize; j += warpSize){
		if (j+thrdIdx < segSize) {
			//			s_ref[sharedBase + ((outCntLT + thrdIdx) & sharedAddrMask)] = ref = refin[thrdIdx];
			ref = refin[thrdIdx];

			// Do the compare
			KdCoord val = coords[ref*dim+p];
			cmp = superKeyComparePD(val, divVal, coords+ref*dim, coords+divRef*dim, p, dim);
			// First check for compare failure
		} else {
			cmp = 0; // Use cmp == 0 to exclude data outside the segment
		}
		refin += warpSize;
		// Write out the less than indices
		uint shflMask = __ballot(cmp<0);
		if (cmp < 0) {
			// Calculate the address which is determined by the number of kept values less than this thread.
			sint wrtIdx = __popc(shflMask & maskGEme);
			s_refLT[sharedBase + ((outCntLT + wrtIdx) & sharedAddrMask)] = ref;
		}
		// Update the output counter but keep an old value so it's known where to write the output.
		oldOutCntLT = outCntLT;
		outCntLT += __popc(shflMask);
		// If the write spilled into the other buffer in shared memory write buffer indicated by old count.
		if (((oldOutCntLT ^ outCntLT) & warpSize) != 0) {
			refoutLT[(oldOutCntLT & ~(warpSize-1)) + thrdIdx] = s_refLT[sharedBase + (oldOutCntLT & warpSize) + thrdIdx];
		}
		// Write out the greater than values
		shflMask = __ballot(cmp>0);
		if (cmp > 0) {
			// Calculate the address which is determined by the number of kept values less than this thread.
			sint wrtIdx = __popc(shflMask & maskGEme);
			s_refGT[sharedBase + ((outCntGT + wrtIdx) & sharedAddrMask)] = ref;
		}
		// Update the output counter but keep an old value so it's known where to write the output.
		oldOutCntGT = outCntGT;
		outCntGT += __popc(shflMask);
		// If the write spilled into the other buffer in shared memory write buffer indicated by old count.
		if (((oldOutCntGT ^ outCntGT) & warpSize) != 0) {
			refoutGT[(oldOutCntGT & ~(warpSize-1)) + thrdIdx] = s_refGT[sharedBase + (oldOutCntGT & warpSize) + thrdIdx];
		}
	}
	// Write out the final LT buffer
	if ((outCntLT & (warpSize-1)) > thrdIdx) {
		refoutLT[(outCntLT & ~(warpSize-1)) + thrdIdx] = s_refLT[sharedBase + (outCntLT & warpSize) + thrdIdx];
	}
	// write out the final GT buffer
	if ((outCntGT & (warpSize-1)) > thrdIdx) {
		refoutGT[(outCntGT & ~(warpSize-1)) + thrdIdx] = s_refGT[sharedBase + (outCntGT & warpSize) + thrdIdx];
	}

	// And finally store the number of LT writes that were done by this warp
	if (thrdIdx == 0 && segLengthsLT != NULL) segLengthsLT[warpIndex] = outCntLT;
	// And finally store the number of GT writes that were done by this warp
	if (thrdIdx == 0 && segLengthsGT != NULL) segLengthsGT[warpIndex] = outCntGT;
}

__global__ void cuPartitionLWTP( KdNode kdNodes[], const __restrict__ KdCoord coords[],
		refIdx_t refoutx[], refIdx_t refinx[], refIdx_t refp[],
		const sint p, const sint dim,
		refIdx_t midRefs[], refIdx_t lastMidRefs[],
		sint startIn, sint endIn, const sint level, const sint logNumWarps)
{
	uint pos = (blockIdx.x * blockDim.x + threadIdx.x); // This thread's position in all threads
	uint allWarps = gridDim.x * blockDim.x / warpSize;  // Total number of warps started
	uint thrdIdx = (pos & (warpSize-1));                // this threads position in the warp.
	uint warpIndex = ((pos - thrdIdx)/warpSize);        // this warps position in all threads
	uint loopLevels = level-logNumWarps;                // log of the nuber of iterations to be done

	// This first loop iterates over the partition regions when there are more partion regiions then thread.
	// Note that if the there are more warps than partition regions (level < logNumWarps) the iteration count will be 0
	// and start and end will be untouched.
	for (uint loop = 0; loop < (1<<loopLevels); loop++) {
		uint start = startIn;                               // local copy of start
		uint end = endIn;                                   // local copy of end
		uint mid;                                           // mid between start and end

		// This loop determines the start and end of the current iteration over the partitions.
		for (uint k = 1;  k <= loopLevels; k++) {
			mid = start + ((end - start)>>1);
			if (loop & (1 << (loopLevels - k ))) {
				start = mid + 1;
			} else {
				end = mid -1;
			}
		}

		// Now calculate the start and end  end and mid using the iterative method for this warps partition segment.
		for (uint i = 0;  i < logNumWarps; i++) {
			mid = start + ((end - start)>>1);
			if (warpIndex & (allWarps >> (i+1))) {
				start = mid + 1;
			} else {
				end = mid -1;
			}
		}
		mid = start + ((end - start)>>1);
		// Calculate the size of the partition segment that this warp will partition.
		sint partSize = end - start + 1; // number of reference to partition
		// get the reference to the coordinate that will be partitioned against.
		refIdx_t midRef = refp[mid];
		cuSinglePartition( coords, // pointer to coordinate array
				refoutx+start, // pointer to the beginning of the output ref array for this subdivision
				refoutx+mid+1, // pointer to the beginning of the output ref array for this subdivision
				refinx+start, // pointer to the beginning of the input ref array for this subdivision
				midRef, // reference to coordinate against which spitting the arrays will be partitioned
				p, dim, // which dimension is being used for partitioning and the number of dimensions
				partSize,// The size of the segment that each warp will partition
				NULL, // pointer to where the resulting segment lengths for this subdivision will be put
				NULL, // pointer to where the resulting segment lengths for this subdivision will be put
				partSize,   // total length for all partitions.  This bounds the partition so no overflow.
				1 // number of warps being applied to partitioning this subdivision
		);

		// if this thread is the 0th thread and this warp is starting warp in a warp group, save off the mid point.
		if (thrdIdx == 0 ) {
			uint mra = warpIndex+loop*allWarps;
			midRefs[mra] = midRef;
			if (lastMidRefs != NULL) {
				if (mra & 1) { // odd or even?
					kdNodes[lastMidRefs[mra>>1]].gtChild = midRef;
				} else {
					kdNodes[lastMidRefs[mra>>1]].ltChild = midRef;
				}
			}
		}
	}
}

__global__ void cuPartition( KdNode kdNodes[], const __restrict__ KdCoord coords[],
		refIdx_t refoutxLT[], refIdx_t refoutxGT[],
		refIdx_t refinx[], refIdx_t refp[],
		const sint p, const sint dim,
		uint segLengthsLT[], uint segLengthsGT[],
		refIdx_t midRefs[], refIdx_t lastMidRefs[],
		const sint startIn, const sint endIn, const sint level)
{
	uint pos = (blockIdx.x * blockDim.x + threadIdx.x);
	uint allWarps = gridDim.x * blockDim.x / warpSize;
	uint thrdIdx = (pos & (warpSize-1));
	uint warpIndex = ((pos - thrdIdx)/warpSize);

	uint start = startIn;
	uint end = endIn;
	uint mid;

	for (uint i = 0;  i < level; i++) {
		mid = start + ((end - start)>>1);
		if (warpIndex & (allWarps >> (i+1))) {
			start = mid + 1;
		} else {
			end = mid -1;
		}
	}
	mid = start + ((end - start)>>1);
	sint partSize = end-start+1;
	uint segSize = (partSize + (allWarps>>level) - 1) / (allWarps>>level);
	refIdx_t midRef = refp[mid];
	cuSinglePartition( coords, // pointer to coordinate array
			refoutxLT+start, // pointer to the beginning of the output ref array for this subdivision
			refoutxGT+start, // pointer to the beginning of the output ref array for this subdivision
			refinx+start, // pointer to the beginning of the input ref array for this subdivision
			midRef, // reference to coordinate against which spitting the arrays will be partitioned
			p, dim, 	// which dimension is being used for partitioning and the number of dimensions
			segSize,	// The size of the segment that each warp will partition
			segLengthsLT+(warpIndex & ~((allWarps >> level) - 1)), // pointer to where the resulting segment lengths for this subdivision will be put
			segLengthsGT+(warpIndex & ~((allWarps >> level) - 1)), // pointer to where the resulting segment lengths for this subdivision will be put
			partSize,   // total length of the partition for all warps.
			(allWarps>>level) // number of warps being applied to partitioning this subdivision
	);

	// if this thread is the 0th thread and this warp is starting warp in a warp group, save off the mid point.
	//    if (thrdIdx == 0 && (warpIndex & ((allWarps >> (level+1)) - 1)) == 0) midRefs[warpIndex >> (level-1)] = midRef;
	if (thrdIdx == 0 ){
		uint mra = warpIndex/(allWarps>>level);
		midRefs[mra] = midRef;
		if (lastMidRefs != NULL) {
			if (mra & 1) { // odd or even?
				kdNodes[lastMidRefs[mra>>1]].gtChild = midRef;
			} else {
				kdNodes[lastMidRefs[mra>>1]].ltChild = midRef;
			}
		}
	}
}

__global__ void cuPartitionLast(KdNode kdNodes[], refIdx_t refp[], refIdx_t midRefs[],	refIdx_t lastMidRefs[],
		const sint startIn, const sint endIn, const sint level)
{
	uint pos = (blockIdx.x * blockDim.x + threadIdx.x);
	uint allWarps = gridDim.x * blockDim.x;

	uint start = startIn;
	uint end = endIn;
	uint mid;
	refIdx_t midRef = -1;

	for (uint i = 0;  i < level; i++) {
		mid = start + ((end - start)>>1);
		if (pos & (allWarps >> (i+1))) {
			start = mid + 1;
		} else {
			end = mid -1;
		}
	}
	if (end - start > 2){
		// set an error condition.  Indicates that not enough partition loops were done.
		d_partitionError = PART_FINISH_DELTA_TOO_LARGE;
	} else if (end - start == 2) {
		mid = start + ((end - start)>>1);
		midRef = refp[mid];
		kdNodes[midRef].gtChild = refp[end];
		kdNodes[midRef].ltChild = refp[start];
	} else if (end - start == 1) {
		midRef = refp[start];
		kdNodes[midRef].gtChild = refp[end];
	} else if (end - start == 0) {
		midRef = refp[start];
	}
	if (midRef != -1){
		midRefs[pos] = midRef;
		if (pos & 1) { // odd or even?
			kdNodes[lastMidRefs[pos>>1]].gtChild = midRef;
		} else {
			kdNodes[lastMidRefs[pos>>1]].ltChild = midRef;
		}
	}
}

void Gpu::initBuildKdTree() {
	uint numWarps = numBlocks*numThreads/32;
#pragma omp critical (launchLock)
	{
		setDevice();
		// Create the array that stores the length of each treads segment length
		checkCudaErrors(cudaMalloc((void **)&d_segLengthsLT, numWarps * sizeof(uint)));
		checkCudaErrors(cudaMalloc((void **)&d_segLengthsGT, numWarps * sizeof(uint)));
		// Allocate the arrays to store the midpoint references for this level
		checkCudaErrors(cudaMalloc((void **)&d_midRefs[0], num * sizeof(refIdx_t)));
		checkCudaErrors(cudaMalloc((void **)&d_midRefs[1], num * sizeof(refIdx_t)));
	}
}
void Gpu::closeBuildKdTree() {
	syncGPU();
#pragma omp critical (launchLock)
	{
		setDevice();
		// Free the array that stores the length of each treads segment length
		checkCudaErrors(cudaFree(d_segLengthsLT));
		checkCudaErrors(cudaFree(d_segLengthsGT));
		// Free the arrays to store the midpoint references for this level
		checkCudaErrors(cudaFree(d_midRefs[0]));
		checkCudaErrors(cudaFree(d_midRefs[1]));
	}
}


void Gpu::partitionDim(KdNode d_kdNodes[], const KdCoord d_coords[], refIdx_t* l_references[],
		const sint p, const sint dim, const sint numTuples, const sint level, const sint numThreads) {

	uint numWarps = numThreads/32;
	uint logNumWarps = (uint)std::log2((float)numWarps);
	uint logNumTuples = (uint)ceil(std::log2((float)numTuples));
	// This portion sets up the tread and block size to work with small numbers of thread
	// This is only useful for debug situations.
	sint numBlocks;
	sint numThrdPerBlk;
	if (numThreads >= SHARED_SIZE_LIMIT/2) {
		numBlocks = numThreads/(SHARED_SIZE_LIMIT/2);
		numThrdPerBlk = SHARED_SIZE_LIMIT/2;
	} else {
		numBlocks = 1;
		numThrdPerBlk = numThreads;
	}

	refIdx_t* thisMidRefs = d_midRefs[level % 2]; // Find out if this is an odd or even level
	refIdx_t* lastMidRefs = d_midRefs[(level-1) % 2]; // Find out if this is an odd or even level
	if (level == 0) {
		lastMidRefs = NULL;  // On the first pass null out the pointer to the last level because there isn't one.
	}

	//#define PRINT_TIME
#ifdef PRINT_TIME
	float time;
	cudaEvent_t t_start, t_stop;
	checkCudaErrors(cudaEventCreate(&t_start));
	checkCudaErrors(cudaEventCreate(&t_stop));

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventRecord(t_start));
#endif

	// Create pointers to the array that stores the length of each treads segment length used in
	// the compaction part of the partitioning functions.  Only needed when there are there are more
	// warps than there are partition segments.  Need 1 array for GT side, 1 array for LT side.

	// When the number of partitions to be performed is less that the number of
	// warps, the partition kernel will apply multiple warps to each partition
	// This is the case where the segLength arrays are needed
	sint loopLevels;
	sint remLevels;
	if (level < logNumWarps){
		loopLevels = 0;
		remLevels = level;


		sint start = 0;
		sint end = numTuples-1;

		for (sint thisDim = 1;  thisDim < dim; thisDim++) {  // Partition every dimension except the p partition.
			sint r = thisDim + p;
			r = (r >= dim) ? r-dim : r;

#pragma omp critical (launchLock)
			{
				setDevice();
				cuPartition<<<numBlocks, numThrdPerBlk, 0, stream>>>(d_kdNodes, d_coords,                   //pointer to coordinates
						l_references[dim], l_references[dim+1], // pointers to the LT and GT partition output arrays
						l_references[r], l_references[p],       // pointers to the partitioned and primary array
						p, dim,                                // axis and number of dimentions
						d_segLengthsLT, d_segLengthsGT,        // used byt the partition kernal to store segment sizes 
						thisMidRefs, 							// array of this midpoint refrences
						lastMidRefs, 							// array of last midpoint refrences
						start, end, remLevels);                // start and end of the data and sub level.
				checkCudaErrors(cudaGetLastError());
				// Do the copy to close up the gaps, lt to the lower half, gt to the upper half
				cuPartitionRemoveGaps<<<numBlocks, numThrdPerBlk, 0, stream>>>(l_references[r], l_references[dim], l_references[dim+1],
						d_segLengthsLT, d_segLengthsGT, start, end, remLevels);
			}
		}
	} else {
		if ((logNumTuples - level > 5)) {
			loopLevels = (level-logNumWarps);
			remLevels = logNumWarps;

			loopLevels = 0;
			remLevels = level;
			for (sint loop = 0; loop < (1<<loopLevels); loop++) {
				sint start = 0;
				sint end = numTuples-1;
				uint mid;
				for (int k=1; k<=loopLevels; k++) {
					mid = start + (end - start)/2;
					if (loop & (1 << loopLevels-k))
						start = mid + 1;
					else
						end = mid - 1;
				}

				for (sint thisDim = 1;  thisDim < dim; thisDim++) {  // partition ever dimension except the p partition.
					sint r = thisDim + p;
					r = (r >= dim) ? r-dim : r;
#pragma omp critical (launchLock)
					{
						setDevice();
						cuPartitionLWTP<<<numBlocks, numThrdPerBlk, 0, stream>>>(d_kdNodes, d_coords,                     //pointer to coordinates
								l_references[dim],                      // pointers to the LT and GT partition output arrays
								l_references[r], l_references[p],       // pointers to the partitioned and primary array
								p, dim,                                 // axis and number of dimensions
								thisMidRefs+loop*numWarps, 				// array of this midpoint references
								lastMidRefs+loop*numWarps/2, 			// array of last midpoint references
								start, end, remLevels, logNumWarps);    // start and end of the data and sub level.
						checkCudaErrors(cudaGetLastError());
						// do the copy to close up the gaps, lt to the lower half, gt to the upper half
						cuCopyRef<<<numBlocks, numThrdPerBlk, 0, stream>>>(l_references[r]+start, l_references[dim]+start, end - start + 1);
					}
				}
			}
		} else {
#define CHECK_FOR_ERRORS
#ifdef CHECK_FOR_ERRORS
			sint partitionError = 0;
			cudaMemcpyToSymbol(d_partitionError,
					&partitionError,
					sizeof(partitionError),
					0,cudaMemcpyHostToDevice);
#endif
			sint logSubWarpSize = logNumTuples - level; // Should never be bigger than 32
			sint logNumSubWarps = logNumWarps + 5 - logSubWarpSize;
			sint start = 0;
			sint end = numTuples-1;
			for (sint thisDim = 1;  thisDim < dim; thisDim++) {  // Partition ever dimension except the p partition.
				sint r = thisDim + p;
				r = (r >= dim) ? r-dim : r;
#pragma omp critical (launchLock)
				{
					setDevice();
					cuPartitionShort<<<numBlocks, numThrdPerBlk, 0, stream>>>(d_kdNodes, d_coords,  //pointer to coordinates
							l_references[dim],                      // pointers to the LT and GT partition output arrays
							l_references[r], l_references[p],       // pointers to the partitioned and primary array
							p, dim,                                 // axis and number of dimentions
							thisMidRefs,                				// array of this midpoint refrences
							lastMidRefs,           					// array of last midpoint refrences
							start, end,                               // start and end of the data
							level, logNumSubWarps, logSubWarpSize);    // sub level.
					checkCudaErrors(cudaGetLastError());
					// Do the copy to close up the gaps, lt to the lower half, gt to the upper half
					cuCopyRef<<<numBlocks, numThrdPerBlk, 0, stream>>>(l_references[r]+start, l_references[dim]+start, end - start + 1);
				}
				checkCudaErrors(cudaGetLastError());
#ifdef CHECK_FOR_ERRORS
				cudaMemcpyFromSymbolAsync(&partitionError,
						d_partitionError,
						sizeof(partitionError),
						0,cudaMemcpyDeviceToHost, stream);
				if (partitionError == PART_SIZE_GT_SUB_PART_SIZE ) {
					cout << "Error in partition size vs sub warp size on level " << level << endl;
					exit(1);
				}
#endif
			}
		}
	}

#ifdef PRINT_TIME
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventRecord(t_stop));
	checkCudaErrors(cudaEventSynchronize(t_stop));
	checkCudaErrors(cudaEventElapsedTime(&time, t_start, t_stop));
	printf ("Partition took %f seconds\n",time/1000.0);
	checkCudaErrors(cudaEventDestroy(t_start));
	checkCudaErrors(cudaEventDestroy(t_stop));
#endif

	// Get the mid values back but only on the first dimension processed of level 0.  This will be the root node
	if (level == 0) {
		checkCudaErrors(cudaMemcpyAsync(&rootNode, d_midRefs[0], sizeof(refIdx_t), cudaMemcpyDeviceToHost, stream));
	}

	if (-1 == rootNode) {
		cout << " Build Tree Error:  Failure in assembly" << endl;
	}

	return;
}

void Gpu::partitionDimLast(KdNode d_kdNodes[], const KdCoord coord[], refIdx_t* l_references[],
		const sint p, const sint dim, const sint numTuples, const sint level, const sint numThreads) {

	uint numWarps = numThreads;
	uint logNumWarps = (uint)std::log2((float)numWarps);

	sint loopLevels;
	sint remLevels;
	if (logNumWarps < level){
		loopLevels = (level-logNumWarps);
		remLevels = logNumWarps;
	} else {
		loopLevels = 0;
		remLevels = level;
	}


	sint numBlocks;
	sint numThrdPerBlk;
	if (numThreads >= SHARED_SIZE_LIMIT/2) {
		numBlocks = numThreads/(SHARED_SIZE_LIMIT/2);
		numThrdPerBlk = SHARED_SIZE_LIMIT/2;
	} else {
		numBlocks = 1;
		numThrdPerBlk = numThreads;
	}

	refIdx_t* thisMidRefs = d_midRefs[level % 2]; // Find out id this is an odd or even level
	refIdx_t* lastMidRefs = d_midRefs[(level-1) % 2]; // Find out id this is an odd or even level

#ifdef PRINT_TIME
	float time;
	cudaEvent_t t_start, t_stop;
	checkCudaErrors(cudaEventCreate(&t_start));
	checkCudaErrors(cudaEventCreate(&t_stop));

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventRecord(t_start));
#endif

	for (sint loop = 0; loop < (1<<loopLevels); loop++) {
		sint start = 0;
		sint end = numTuples-1;
		sint mid;
		for (sint k=1; k<=loopLevels; k++) {
			mid = start + (end - start)/2;
			if (loop & (1 << loopLevels-k))
				start = mid + 1;
			else
				end = mid - 1;
		}

		for (sint thisDim = 1;  thisDim < 2; thisDim++) {  // Partition ever dimension except the p partition.
			sint r = thisDim + p;
			r = (r >= dim) ? r-dim : r;
#ifdef CHECK_FOR_ERRORS
			sint partitionError = 0;
			cudaMemcpyToSymbol(d_partitionError, &partitionError,
					sizeof(partitionError), 0,cudaMemcpyHostToDevice);
#endif
#pragma omp critical (launchLock)
			{
				setDevice();
				cuPartitionLast<<<numBlocks, numThrdPerBlk, 0, stream>>>(d_kdNodes, // pointer to kdnode array.
						l_references[p],  // Reference array for primary
						thisMidRefs+loop*numThreads, // mid reference array for current level
						lastMidRefs+loop*numThreads/2, // mid reference array for last level
						start, end, remLevels); // Address range and more levels.
				checkCudaErrors(cudaGetLastError());
			}
#ifdef CHECK_FOR_ERRORS
			cudaMemcpyFromSymbol(&partitionError, d_partitionError,
					sizeof(partitionError), 0,cudaMemcpyDeviceToHost);
			if (partitionError == PART_FINISH_DELTA_TOO_LARGE ) {
				cout << "Error in last partition pass.  Probably due to insufficient number of partiion passes, level = " << level << endl;
				exit(1);
			}
#endif
		}
	}
	//	  checkCudaErrors(cudaGetLastError());
#ifdef PRINT_TIME
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventRecord(t_stop));
	checkCudaErrors(cudaEventSynchronize(t_stop));
	checkCudaErrors(cudaEventElapsedTime(&time, t_start, t_stop));
	printf ("Partition took %f seconds\n",time/1000.0);
	checkCudaErrors(cudaEventDestroy(t_start));
	checkCudaErrors(cudaEventDestroy(t_stop));
#endif


	return;
}

uint Gpu::copyRef(refIdx_t refout[], refIdx_t refin[], uint numTuples, sint numThreads){
	// This portion sets up the tread and block size to work with small numbers of thread
	// This is only useful for debug situations.
	sint numBlocks;
	sint numThrdPerBlk;
	if (numThreads >= SHARED_SIZE_LIMIT/2) {
		numBlocks = numThreads/(SHARED_SIZE_LIMIT/2);
		numThrdPerBlk = SHARED_SIZE_LIMIT/2;
	} else {
		numBlocks = 1;
		numThrdPerBlk = numThreads;
	}
	cuCopyRef<<<numBlocks, numThrdPerBlk, 0, stream>>>(refout, refin, numTuples);
	return 0;
}


