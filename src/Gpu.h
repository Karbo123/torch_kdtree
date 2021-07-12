//
//  Gpu.h
//  KdTreeGPUsms
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


#ifndef gpu_Gpu_h
#define gpu_Gpu_h

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <helper_cuda.h>


using namespace std;

#include "KdNode.h"

struct NodeCoordIndices { refIdx_t node_index; refIdx_t coord_index; };
struct StartEndIndices { refIdx_t start_index; refIdx_t end_index; };
struct FrontEndIndices { refIdx_t front_index; refIdx_t end_index; };

class Gpu {
	// Gpu class constants;
	static const uint MAX_THREADS = 1024;
	static const uint MAX_BLOCKS = 1024;
	static const sint CUDA_QUEUE_MAX = 8; // the assumed max size of one queue

public:
	// These are the API methods used outside the class.  They hide any details about the GPUs from the main program.
	static Gpu* gpuSetup(int threads, int blocks, int dim, int gpuid, cudaStream_t torch_stream);
	void        initializeKdNodesArray(KdCoord coordinates[], const sint numTuples, const sint dim);
	void        mergeSort(sint end[], const sint numTuples, const sint dim);
	refIdx_t    buildKdTree(KdNode kdNodes[], const sint numTuples, const sint dim);
	sint        verifyKdTree(KdNode kdNodes[], const sint root, const sint dim, const sint numTuples);
	void        getKdTreeResults(KdNode kdNodes[], const sint numTuples);
	int         getNumThreads() { return this->numThreads; }
	int         getNumBlocks() { return this->numBlocks; }

	// Device specific variables
private:
	sint 		numThreads; 	// Constant value holding the number of threads
	sint 		numBlocks; 		// Constant value holding the number of blocks
	sint 		devID; 			// The GPU device we are talking to.
	refIdx_t** 	d_references;	// Pointer to array of pointers to reference arrays
	KdCoord**  	d_values;		// Pointer to array of pointers to value arrays
	KdCoord* 	d_coord;		// Pointer to coordinate array @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	KdNode* 	d_kdNodes;		// Pointer to array of KdNodes @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	sint* 		d_end;         // Pointer to array of end values in th GPU
	cudaStream_t stream;       // Cuda stream for this GPU
	cudaEvent_t syncEvent;   	// Cuda sync events
	cudaEvent_t start, stop;   // Cuda Timer events
	uint 		dimen;	       // Number of dimensions
	uint 		num;           // Number of tuples or points
	refIdx_t    rootNode;      // Store the root node here so all partitionDim rouotines can get to it. @@@@@@@@@@@@@@@@@
	sint* d_partitionError;
	uint* d_verifyKdTreeError;
	uint* d_pivot;
	uint* d_removeDupsCount;
	sint* d_removeDupsError;
	sint* d_removeDupsErrorAdr;

private:
	// for CUDA querying
	NodeCoordIndices* d_index_temp;
	NodeCoordIndices* d_index_down;
	NodeCoordIndices* d_index_up;
	sint* d_num_temp;
	sint* d_num_down;
	sint* d_num_up;
	StartEndIndices* d_queue;
	FrontEndIndices* d_queue_frontend;
	sint num_of_points;

public:
	// Constructor
	Gpu(int threads, int blocks, int dim, int dev, cudaStream_t torch_stream) {
		devID = dev;
		numThreads = threads;
		if (numThreads>MAX_THREADS) numThreads = 0;
		numBlocks = blocks;
		if (numBlocks>MAX_BLOCKS) numBlocks = 0;
		d_references = new refIdx_t*[dim+2];
		for (sint i = 0;  i < dim + 2; i++) d_references[i] = NULL;
		d_values = new KdCoord*[dim+2];
		for (sint i = 0;  i < dim + 2; i++) d_values[i] = NULL;
		d_coord = NULL;
		d_kdNodes = NULL;
		d_end = NULL;
		d_mpi = NULL;
		d_segLengthsLT = NULL;
		d_segLengthsGT = NULL;
		d_midRefs[0] = NULL;
		d_midRefs[1] = NULL;

		setDevice();
		// checkCudaErrors(cudaStreamCreate(&stream));
		stream = torch_stream;
		
		checkCudaErrors(cudaMalloc((void**)&d_partitionError, sizeof(sint))); 
		checkCudaErrors(cudaMalloc((void**)&d_verifyKdTreeError, sizeof(uint))); 
		checkCudaErrors(cudaMalloc((void**)&d_pivot, sizeof(uint))); 
		checkCudaErrors(cudaMalloc((void**)&d_removeDupsCount, sizeof(uint))); 
		checkCudaErrors(cudaMalloc((void**)&d_removeDupsError, sizeof(sint))); 
		checkCudaErrors(cudaMalloc((void**)&d_removeDupsErrorAdr, sizeof(sint))); 

		d_index_temp = nullptr;
		d_index_down = nullptr;
		d_index_up   = nullptr;
		d_num_temp = nullptr;
		d_num_down = nullptr;
		d_num_up   = nullptr;
		d_queue = nullptr;
		d_queue_frontend = nullptr;
		num_of_points = 0;

		checkCudaErrors(cudaEventCreate(&syncEvent));
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		dimen = dim;
		num = 0;
	}


	~Gpu(){
		setDevice();
		if (d_references != NULL) {
			for (sint i = 0; i < dimen+1; i++)
				checkCudaErrors(cudaFree(d_references[i]));
		}
		free(d_references);
		if (d_values != NULL) {
			for (sint i = 0; i < dimen+1; i++)
				checkCudaErrors(cudaFree(d_values[i]));
		}
		free(d_values);
		if (d_coord != NULL)
			checkCudaErrors(cudaFree(d_coord));
		if (d_coord != NULL)
			checkCudaErrors(cudaFree(d_kdNodes));
		if (d_end != NULL)
			checkCudaErrors(cudaFree(d_end));
		if (d_partitionError != NULL)
			checkCudaErrors(cudaFree(d_partitionError));
		if (d_verifyKdTreeError != NULL)
			checkCudaErrors(cudaFree(d_verifyKdTreeError));
		if (d_pivot != NULL)
			checkCudaErrors(cudaFree(d_pivot));
		if (d_removeDupsCount != NULL)
			checkCudaErrors(cudaFree(d_removeDupsCount));
		if (d_removeDupsError != NULL)
			checkCudaErrors(cudaFree(d_removeDupsError));
		if (d_removeDupsErrorAdr != NULL)
			checkCudaErrors(cudaFree(d_removeDupsErrorAdr));

		// free memory for query
		DestroyQueryMem();

		checkCudaErrors(cudaEventDestroy(start));
		checkCudaErrors(cudaEventDestroy(stop));
		// checkCudaErrors(cudaStreamDestroy(stream));
		num = 0;
	}

private:
	// These are the per GPU methods.  They implemented in Gpu.cu
	void inline setDevice() {checkCudaErrors(cudaSetDevice(devID));}
	void initializeKdNodesArrayGPU(const KdCoord coord[], const int numTuples, const int dim);
	void initializeReferenceGPU(const sint numTuples, const sint p, const sint dim);
	void mergeSortRangeGPU(const sint start, const sint num, const sint from, const sint to,
			const sint p, const sint dim);
	sint removeDuplicatesGPU(const sint start, const sint num, const sint from, const sint to,
			const sint p, const sint dim, Gpu* otherGpu = NULL, sint otherNum = 0);
	refIdx_t buildKdTreeGPU(const sint numTuples, const int startP, const sint dim);
	void getKdNodesFromGPU(KdNode kdNodes[], const sint numTuples);
	void getReferenceFromGPU(refIdx_t reference[], const sint p, uint numTuples);
	void copyRefValGPU(sint start, sint num, sint from, sint to);
	void copyRefGPU(sint start, sint num, sint from, sint to);
	sint balancedSwapGPU(sint start, sint num, sint from, sint p, sint dim, Gpu* otherGpu);
	void swapMergeGPU(sint start, sint num, sint from, sint to, sint mergePoint,
			const sint p, const sint dim);
	void getCoordinatesFromGPU(KdCoord coord[],  const uint numTuples,  const sint dim);
	void fillMemGPU(uint* d_pntr, const uint val, const uint num);
	void fillMemGPU(sint* d_pntr, const sint val, const uint num);
	inline void syncGPU() { checkCudaErrors(cudaStreamSynchronize(stream));}

private: // These are the methods specific mergeSort
	uint *d_RanksA, *d_RanksB, *d_LimitsA, *d_LimitsB;
	uint maxSampleCount;
	sint *d_mpi;  //This is where the per partition merge path data will get stored.
	refIdx_t* d_iRef;
	KdCoord* d_iVal;

	void mergeSortShared(
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
	);

	void generateSampleRanks(
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
	);

	void mergeRanksAndIndices(
			uint *d_LimitsA,
			uint *d_LimitsB,
			uint *d_RanksA,
			uint *d_RanksB,
			uint stride,
			uint N
	);

	void mergeElementaryInterRefs(
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
	);

	void initMergeSortSmpl(uint N);
	void closeMergeSortSmpl();
	void mergeSortSmpl(
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
	);
	uint balancedSwap(KdCoord* coordA, KdCoord* valA, refIdx_t* refA,
			KdCoord* coordB, KdCoord *valB, refIdx_t* refB,
			sint sortDir, sint p, sint dim, uint NperG, sint numThreads);

	void mergeSwap(KdCoord d_coord[], KdCoord d_valSrc[], refIdx_t d_refSrc[],
			KdCoord d_valDst[], refIdx_t d_refDst[],
			sint mergePnt, sint p, sint dim,  uint N, sint numThreads);

private: // These are the methods specific removeDupes
	uint copyRefVal(KdCoord valout[], refIdx_t refout[], KdCoord valin[], refIdx_t refin[], uint numTuples, sint numThreads);

	uint removeDups(KdCoord coords[], KdCoord val[], refIdx_t ref[], KdCoord valtmp[], refIdx_t reftmp[],
			KdCoord valin[], refIdx_t refin[], KdCoord otherCoord[], refIdx_t *otherRef,
			const sint p, const sint dim, const sint  numTuples, sint numThreads);

private: // These are the methods specific buildKdTree
	uint* d_segLengthsLT;  // When doing a segmented partition, pointers to the segment length arrays are stored here
	uint* d_segLengthsGT;
	refIdx_t* d_midRefs[2];

	void initBuildKdTree();
	void closeBuildKdTree();
	void partitionDim(KdNode d_kdNodes[], const KdCoord d_coords[], refIdx_t* l_references[],
			const sint p, const sint dim, const sint numTuples, const sint level, const sint numThreads);

	void partitionDimLast(KdNode d_kdNodes[], const KdCoord coord[], refIdx_t* l_references[],
			const sint p, const sint dim, const sint numTuples, const sint level, const sint numThreads);

	uint copyRef(refIdx_t refout[], refIdx_t refin[], uint numTuples, sint numThreads);

private: // These are the methods specific verifyKdTree
	// Pointer to the array for summing the node counts
	int* d_sums;
	void initVerifyKdTree();
	void closeVerifyKdTree();
	int  verifyKdTreeGPU(const sint root, const sint pstart, const sint dim, const sint numTuples);


	/*
	 * The createKdTree function performs the necessary initialization then calls the buildKdTree function.
	 *
	 * calling parameters:
	 *
	 * coordinates - a array  of coordinates ie (x, y, z, w...) tuples
	 * numDimensions - the number of dimensions
	 *
	 * returns: a KdNode pointer to the root of the k-d tree
	 */
public:
	static KdNode *createKdTree(Gpu* device, KdNode kdNodes[], KdCoord coordinates[],  const sint numDimensions, const sint numTuples);


//////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////
private: // functions for query on CUDA
	void InitQueryMem(sint _num_of_points)
	{
		if (_num_of_points > num_of_points) // memory not enough
		{
			DestroyQueryMem();
			checkCudaErrors(cudaMalloc((void**)&d_index_temp, sizeof(NodeCoordIndices) * _num_of_points));
			checkCudaErrors(cudaMalloc((void**)&d_index_down, sizeof(NodeCoordIndices) * _num_of_points));
			checkCudaErrors(cudaMalloc((void**)&d_index_up, sizeof(NodeCoordIndices) * _num_of_points));
			checkCudaErrors(cudaMalloc((void**)&d_num_temp, sizeof(sint)));
			checkCudaErrors(cudaMalloc((void**)&d_num_down, sizeof(sint)));
			checkCudaErrors(cudaMalloc((void**)&d_num_up, sizeof(sint)));
			checkCudaErrors(cudaMalloc((void**)&d_queue, sizeof(StartEndIndices) * _num_of_points * CUDA_QUEUE_MAX));
			checkCudaErrors(cudaMalloc((void**)&d_queue_frontend, sizeof(FrontEndIndices) * _num_of_points));
		}
		num_of_points = _num_of_points; // num of querying points
	}
	void DestroyQueryMem()
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

public:
	void InitSearch(sint _num_of_points);

};


#endif
