//
//  Gpu.cu
//  This file contains the definition of the Gpu class which provides the GPU
//  API for the GPU fuctions requred to build the Kd tree.  
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

/********************************************************************************
/* DBUG defines
/********************************************************************************/

//#define FAKE_TWO // runs the Multi-GPU code on a single GPU

#include <limits>
#include <cuda_runtime.h>
#include <omp.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>  // Helper for shared that are common to CUDA Samples

#include "Gpu.h"
#include "mergeSort_common.h"
#include "removeDups_common.h"
#include "buildKdTree_common.h"



/*
 * cuSuperKeyCompare function performs the compare between coordinates used in the sorting and partitioning
 * functions.  It starts by subtracting the primary coordinate or the pth coordinate and proceeds through each
 * of the coordinates until it finds the non-zero difference.  That difference is returned as the compare result.
 * Inputs
 * a[]  Pointer to the first coordinate
 * b[]  Pointer to the second coordinate
 * p    index to the primary coordinate to compare. p must be less than dim
 * dim  Number of dimensions the coordinate has.
 *
 * Returns a long that is positive if a>b, 0 if equal an negative is a<b
 */

__device__ KdCoord cuSuperKeyCompare(const KdCoord a[], const KdCoord b[], const sint p, const sint dim)
{
	KdCoord diff=0;
	for (sint i = 0; i < dim; i++) {
		sint r = i + p;
		r = (r < dim) ? r : r - dim;
		diff = a[r] - b[r];
		if (diff != 0) {
			break;
		}
	}
	return diff;
}

/*
 * cuSuperKeyCompareFirstDim is a GPU function that performs the same function as cuSuperKeyCompare.
 * But in the case where the calling code has pre-fetched the first dimension or component, the this takes
 * the A and B components as L values and only access the array values if the first components happen to be equal.
 * Inputs
 * ap         first compare component l value
 * bp         first compare component l value
 * *a         a coordinates
 * *b         b coordinates
 * p          index of the first
 * dim        number of dimensions the coordinates have
 *
 * Returns a long that is positive if a>b, 0 if equal an negative is a<b
 */

__device__ KdCoord cuSuperKeyCompareFirstDim(const KdCoord ap, const KdCoord bp, const KdCoord *a, const KdCoord *b, const sint p, const sint dim)
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
 * cuInitializeKnodesArray is a GPU kernel that initializes the array of KdNodes that will eventually
 * be the kdNode tree. Initialization include copying the coordinates from the coordinates array and
 * initing the child node indices to -1 which is the terminal node indicator
 * Inputs
 * kdNodes          Pointer to the array of uninitialized kd nodes.
 * coordinates      Pointer the the array coordinates.
 * numTuples        number of coordinates and kd nodes.
 * dim              dimension of the coordinates
 * numTotalThreads  number of threads being used to do the initing
 */

__global__ void cuInitializeKdNodesArray(KdNode kdNodes[], KdCoord coordinates[], const sint numTuples, const sint dim, sint numTotalThreads){

	sint index = threadIdx.x + blockIdx.x * blockDim.x;

	for (sint i = index; i < (numTuples); i+=numTotalThreads) {
		kdNodes[i].tuple   =  i;
		kdNodes[i].ltChild = -1;
		kdNodes[i].gtChild = -1;
	}
}

/*
 * cuCupyCoordinate is a GPU kernel that copies the pth coordinate from the coordinate array into a 1 dimensional array.
 * The copy is done suche that the value is at the same index as the reference in the d_ref array.
 * Inputs
 * coord[]        pointer to the coordinates array.
 * d_sval[]       pointer to the value array that the data is to be copied too.
 * d_ref[]		  pointer to reference array that it is should match
 * p              sint indicating the coordinate to copy
 * dim            sint indicating dimensions
 * numTuples      sint indicating number of values to copy
 */

__global__ void cuCopyCoordinate(const KdCoord coord[], KdCoord d_sval[], const refIdx_t d_ref[], const sint p, const sint dim, const sint numTuples){

	sint index = threadIdx.x + blockIdx.x * blockDim.x;
	sint stride = blockDim.x * gridDim.x;

	for (sint j = 0+index; j < numTuples; j+=stride) {
		refIdx_t t = d_ref[j];
		d_sval[j] = coord[t*dim+p];
	}
}


/*
 * initializeKdNodesArrayGPU is a Gpu class method that allocated the KdNode array in the GPU and calls the cuInitilizeKdNodesArray.
 * It also copies the coordinates data over to the gpu if it's not there already.
 * Inputs
 * coordinates[]  Pointer to the coordinate data the to be put in the kdTree.  Can be null if the GPU already has the coordinates
 * numTuples      sint indicating number of coordinates and the number of kdNodes to be created
 * dim            number of dimensions.
 */
void Gpu::initializeKdNodesArrayGPU(const KdCoord coordinates[], const sint numTuples, const sint dim){
	// Make this whole fuction critical so that memory allocations will not fail.
#pragma omp critical (launchLock)
	{
		setDevice();
		// First allocate memory for the coordinate array and copy it to the device
		if (d_coord == NULL) {
			checkCudaErrors(cudaMalloc((void **) &d_coord, (numTuples+1)*sizeof(int)*dim)); // Allocate an extra for max coord
			checkCudaErrors(cudaMemcpyAsync(d_coord, coordinates, numTuples*sizeof(int)*dim, cudaMemcpyHostToDevice, stream));
		} else if (d_coord != NULL) {
			throw runtime_error("initializeKdNodesArrayGPU Error: coordinate array already allocated");
		}

		// Add an extra tuple at the end with all max values.
		KdCoord tmp[dim];
		for (int i=0; i<dim; i++){
			tmp[i] = std::numeric_limits<KdCoord>::max();
		}
		checkCudaErrors(cudaMemcpyAsync(d_coord+numTuples*dim, tmp, sizeof(KdCoord)*dim, cudaMemcpyHostToDevice, stream));

		// Then allocate the kdNode Array
		if(d_kdNodes == NULL) {
			checkCudaErrors(cudaMalloc((void **) &d_kdNodes, numTuples*sizeof(KdNode)));
		} else {
			throw runtime_error("InitialzeKdNode Error: kdNodes array already allocated");
		}
		// Call the init routine
		cuInitializeKdNodesArray<<<numBlocks, numThreads, 0, stream>>>(d_kdNodes, d_coord, numTuples, dim, numThreads*numBlocks);
		checkCudaErrors(cudaGetLastError());
	}
}


/*
 * initializeKdNodesArray is a Gpu class static method that allocated amd initializes the KdNode array
 * in all of the GPUs.  It also copies a portion of coordinates data over to each gpu.
 * Inputs
 * kdNodes[]      pointer to a cpu side kdNodes array.  If not null, the GPU side kdNodes array is copied to it.
 *                Normally this should be null
 * coordinates[]  Pointer to the CPU coordinate data.
 * numTuples      sint indicating number of coordinates and the number of kdNodes to be created
 * dim            number of dimensions.
 */
void Gpu::initializeKdNodesArray(KdCoord coordinates[], const sint numTuples, const sint dim){

	if (coordinates == NULL) {
		throw runtime_error("initializeKdNodesArray Error: Expecting coordinates data to send to GPU");
	}
	this->initializeKdNodesArrayGPU(coordinates, numTuples, dim);
}

/*
 * cuFillMem is a Gpu class method that fills a portion of memory with a constant value.
 * Inputs
 * d_pntr           pointer to where the fill should start
 * val              value to fill with
 * numTuples        number of reference array entries
 */
__global__ void cuFillMem(uint d_pntr[], uint val, int num){

	sint index = threadIdx.x + blockIdx.x * blockDim.x;
	sint stride = blockDim.x * gridDim.x;

	for ( sint j = 0+index; j < num; j+=stride) {
		d_pntr[j] = val;
	}
}

/*
 * fillMemGPU is a Gpu class method that fills a portion of memory with a constant value.
 * It calls the cuFillMem kernel to do so.
 * Inputs
 * d_pntr           pointer to where the fill should start
 * val              value to fill with
 * numTuples        number of reference array entries
 */
void Gpu::fillMemGPU(uint* d_pntr, const uint val, const uint num) {
	setDevice();
	if (d_pntr == NULL) {
		cuFillMem<<<numBlocks,numThreads, 0, stream>>>(d_pntr, val, num);
		checkCudaErrors(cudaGetLastError());
	} else {
		throw runtime_error("fillMemGPU Error: device pointer is null");
	}
}

void Gpu::fillMemGPU(sint* d_pntr, const sint val, const uint num) {
	setDevice();
	if (d_pntr == NULL) {
		cuFillMem<<<numBlocks,numThreads, 0, stream>>>((uint*)d_pntr, val, num);
		checkCudaErrors(cudaGetLastError());
	} else {
		throw runtime_error("fillMemGPU Error: device pointer is null");
	}
}

/*
 * cuInitializeRefernece is a GPU kernel that initializes the reference arrays used in sorting and partitioning functions
 * Each array value is set to its index.
 * Inputs
 * reference[]       Pointer to the reference array to be initialized
 * numTuples         Integer indicating the number of elements to init.
 */

__global__ void cuInitializeReference(refIdx_t reference[], sint numTuples){

	sint index = threadIdx.x + blockIdx.x * blockDim.x;
	sint stride = blockDim.x * gridDim.x;

	for ( sint j = 0+index; j < numTuples; j+=stride) {
		reference[j] = (refIdx_t)j;
	}
}


/*
 * initializeReferenceGPU is a Gpu class method that allocates the reference arrays in the GPU and then calls
 * cuInitializeReference kernel to init the arrays.  dim+2 arrays are allocated but on dim arrays are initialized
 * Inputs
 * numTuples        number of reference array entries
 * dim              number of dimensions
 * Outputs
 * references       pointer to an array of pointers of CPU side reference arrays. If not null
                    the initialize arrays are copied back. Normally this should be null.
 */
void Gpu::initializeReferenceGPU(const sint numTuples, const sint p, const sint dim) {
	setDevice();
	for(sint i = 0;  i < dim+2; i++) { // Take care of any null pointer
		if (d_references[i] == NULL) {
			// Allocate the space in the GPU
			checkCudaErrors(cudaMalloc((void **) &d_references[i], numTuples*sizeof(uint)));
		}
	}
	// Now initialize on the first dim arrays
#pragma omp critical (launchLock)
	{
		setDevice();
		cuInitializeReference<<<numBlocks,numThreads, 0, stream>>>(d_references[p], numTuples);
		checkCudaErrors(cudaGetLastError());
	}
}


/*
 * getReferenceFromGPU is a Gpu class method that copies one of the gpu reference arrays
 * bach to the CPU.
 * Inputs
 * numTuples        number of reference array entries
 * p				indicates to copy the data from d_reference[p] array.
 * Outputs
 * reference       pointer to a reference array on the CPU side.
 */

void Gpu::getReferenceFromGPU(refIdx_t* reference, const sint p, uint numTuples){
	setDevice();
	// If references is not null copy the init values back
	if (reference != NULL) {
		checkCudaErrors(cudaMemcpyAsync(reference, d_references[p],  numTuples*sizeof(refIdx_t), cudaMemcpyDeviceToHost, stream));
	}
}

/*
 * getCoordinatesFromGPU is a Gpu class method that copies coordinate arrays
 * bach to the CPU.
 * Inputs
 * numTuples        number of tuples in the coordinate
 * dim				number of dimensions in a tuple.
 * Outputs
 * coord       		pointer to a coordinate array on the CPU side.
 */

void Gpu::getCoordinatesFromGPU(KdCoord* coord,  const uint numTuples,  const sint dim){
	setDevice();
	// If references is not null copy the init values back
	if (coord != NULL) {
		checkCudaErrors(cudaMemcpyAsync(coord, d_coord,  dim *numTuples*sizeof(KdCoord), cudaMemcpyDeviceToHost, stream));
	}
}


/*
 * mergeSortRangeGPU is a Gpu class method that performs the sort on one dimension and in one GPU.
 * Inputs
 * start            integer offset into the references[from] array where of where to start the sort
 * num              number of elements to sort
 * from             index of the references array to sort from
 * to               index of the references array to sort to
 * p                primary coordinate on which the sort will occur
 * dim              number of dimensions
 * Output
 */
void Gpu::mergeSortRangeGPU(const sint start, const sint num, const sint from, const sint to, const sint p, const sint dim){
	setDevice();
	// First check that memory on the GPU has been allocated
	if (d_coord == NULL || d_references[from] == NULL || d_references[to] == NULL) {
		throw runtime_error("mergeSortRangeGPU Error: coordinates or references for are null");
	}

	// Set up refVal and tmpVal arrays
#pragma omp critical (launchLock)
	{
		setDevice();
		if (d_values[from] == NULL){
			checkCudaErrors(cudaMalloc((void **) &d_values[from], num*sizeof(KdCoord)));
			// Copy the coordinate of interest to the refVal array
			cuCopyCoordinate<<<numBlocks,numThreads, 0, stream>>>(d_coord+start*dim, d_values[from], d_references[from], from, dim, num);
		}
		if (d_values[to] == NULL)
			checkCudaErrors(cudaMalloc((void **) &d_values[to], num*sizeof(KdCoord)));
		checkCudaErrors(cudaGetLastError());
	}

	mergeSortSmpl(d_coord,							// coordinate array
			d_values[to], d_references[to]+start,    	// output arrays
			d_iVal, d_iRef, 							// Intermediate arrays
			d_values[from], d_references[from]+start,   // Input arrays
			num, 1, p, dim						// sizes and directions
	);
}

/*
 * removeDuplicatesGPU is a Gpu class method that performs the  duplicates removal for one of
 * the dimensions of the coordinates.
 * Inputs
 * start            integer offset into the references[from] array where of where to start the removal
 * num              number of elements to check for removal
 * from             index of the references array to remove from
 * to               index of the references arry to put the results
 * p                primary coordinate on which the removal will occur
 * dim              number of dimensions
 * otherGpu         pointer to another GPU. This function will compare the last tuple in that
 *                  GPU to the first tuple in this one.
 * otherNum         number of tuples in the other GPU.
 * Output
 */
sint Gpu::removeDuplicatesGPU(const sint start, const sint num, const sint from, const sint to,
		const sint p, const sint dim, Gpu* otherGpu, sint otherNum){
	if (d_values[from] == NULL || d_values[to] == NULL) {
		throw runtime_error("values[from] or values[to] pointer is NULL");
	}
	if (d_references[from] == NULL || d_references[to] == NULL) {
		throw runtime_error("references[from] or references[to] pointer is NULL");
	}
	// get the pointers to the data in the other GPU if required.  This is to remove duplicates across GPU boundaries.
	refIdx_t* otherRef   = (otherGpu == NULL) ? NULL : otherGpu->d_references[from]+otherNum-1;
	KdCoord*  otherCoord = (otherGpu == NULL) ? NULL : otherGpu->d_coord;

	sint end = removeDups(d_coord+start*dim,        	// Coordinate array
			d_values[to], d_references[to]+start,    	// Output arrays
			d_iVal, d_iRef, 							// Intermediate arrays
			d_values[from], d_references[from]+start,   // Input arrays
			otherCoord, otherRef,						// Pointers to data in the other GPU
			p, dim, num,            			    	// sizes
			numBlocks*numThreads         				// threads
	);

	return end;
}

/*
 * copyRefValGPU is a Gpu class method that copies the contents of references[from] to the
 * references[to] array.  Likewise for the values arrays.
 * Note that if the references[to] or the values[to] array pointer is null, 
 * the arrays will be allocated
 * Inputs
 * start            integer offset into the references[from] array where of where to start the copy
 * num              number of elements to check for removal
 * from             index of the references array to remove from
 * to               index of the references array to put the results
 * Output
 */
void Gpu::copyRefValGPU(sint start, sint num, sint from, sint to) {
	setDevice();
	// Check for NULL pointers
	if (d_values[from] == NULL) {
		throw runtime_error("copyRefValGPU Error: values[from] pointer is NULL");
	}
	if (d_references[from] == NULL) {
		throw runtime_error("copyRefValGPU Error: references[from] pointer is NULL");
	}
	if (d_values[to] == NULL)
		checkCudaErrors(cudaMalloc((void **) &d_values[to], num*sizeof(KdCoord)));
	if (d_references[to] == NULL)
		checkCudaErrors(cudaMalloc((void **) &d_references[to], num*sizeof(refIdx_t)));
	// Call the copy function
	copyRefVal(d_values[to], d_references[to]+start,
			d_values[from], d_references[from]+start,
			num, numBlocks*numThreads);
}

/*
 * copyRefGPU is a Gpu class method that copies the contents of references[from] to the
 * references[to] array.
 * Note that if the references[to] array pointer is null, the arrays will be allocated
 * Inputs
 * start            integer offset into the references[from] array where of where to start the removal
 * num              number of elements to copy
 * from             index of the references array to copy from
 * to               index of the references array to put the results
 * Output
 */
void Gpu::copyRefGPU(sint start, sint num, sint from, sint to) {
	this->setDevice();
	// Check for NULL pointers
	if (d_references[from] == NULL) {
		cout << "copyRefGPU Error: references[from] pointer is NULL" << endl;
		exit(1);
	}
	if (d_references[to] == NULL)
		checkCudaErrors(cudaMalloc((void **) &d_references[to], num*sizeof(refIdx_t)));
	// Call the copy function
	copyRef(d_references[to]+start,
			d_references[from]+start,
			num, numBlocks*numThreads);
}

/*
 * balancedSwapGPU is a Gpu class method that is a wrapper around the balancedSwap function in mergSort.cu
 * This function uses just one of the GPUs to swap coordinate data between GPUs such that all of the tuples
 * in the is GPU is less than the tuples in the other GPU.  It is a component of the multi GPU sort function.
 * Inputs
 * start            integer offset into the references[from] array where of where to start the removal
 * num              number of elements to check for swap
 * from             index of the references array to swap from.  Results remain in the reference[from]
 * p                primary coordinate on which the swap compare will occur
 * dim              number of dimensions
 * otherGPU         pointer to another GPU.
 * Return
 * pivot			the index into the reference array on the other GPU below which were swapped with this GPU.
 */

sint Gpu::balancedSwapGPU(sint start, sint num, sint from, sint p, sint dim, Gpu* otherGpu){
	setDevice();
	return balancedSwap(this->d_coord,
			this->d_values[from], this->d_references[from]+start,
			otherGpu->d_coord,
			otherGpu->d_values[from], otherGpu->d_references[from]+start,
			1, p, dim, num, numBlocks*numThreads);
}

/*
 * swapMergeGPU is a Gpu class method that is a wrapper around the mergeSwap function in mergSort.cu
 * After the BalancedSwap function exchanges the coordinates between the two GPU, there remains
 * two independently sorted arrays in each GPU.  This function merges those into a single sorted array.
 * Inputs
 * start            integer offset into the references[from] array where of where start of the lower sorted data
 * num              number of elements to check for swap
 * from             index of the references array to merge from.
 * to               index of the references array to merge to.
 * mergePoint		index of the start of the upper sorted data.
 * p                primary coordinate on which the merge compare will occur
 * dim              number of dimensions
 * Output
 */
void Gpu::swapMergeGPU(sint start, sint num, sint from, sint to, sint mergePoint, const sint p, const sint dim){
	setDevice();
	mergeSwap(d_coord,
			d_values[from],
			d_references[from],
			d_values[to],
			d_references[to],
			mergePoint, p, dim, num, numBlocks*numThreads);
}

/*
 * mergeSort is a static Gpu class method that performs the sort and the duplicates removal for all of
 * the dimensions across all GPUs.
 * If there is 1 gpu, this function does a simple loop through the dimensions, first sorting then
 * removing duplicated.
 * If there are 2 gpus, this function, does a sort of the first dimension on the coordinates in
 * each GPU.  the gpus each halve half of the coordinates.  Then it swaps the necessary tuples
 * between GPUs such all of the tuples in gpu 0 are less than all of the tupes in gpu 1.  Finally
 * it copies the references from dimension 1 into each reference arrys of the other dimensions,
 * sorts those arrays and does a duplicate removal.
 * Inputs
 * numTuples        number of input coordinates and references
 * dim              number of dimensions
 * Output
 * end[]            pointer to an array continuing the number of references after duplicate removal. 
 *                  This method writes to the pth entry in that array.
 */
void Gpu::mergeSort(sint end[], const sint numTuples, const sint dim)
{
	sint NperG = numTuples;
	this->initMergeSortSmpl(numTuples);

	//Get first node
	this->setDevice();
	for (int p=0;  p<dim; p++) {
		this->initializeReferenceGPU(NperG, p, dim);
		this->mergeSortRangeGPU(0, NperG, p, dim, p, dim);
		end[p] = this->removeDuplicatesGPU(0, NperG, dim, p, p, dim);
	}

	this->num = end[0];
	
	this->syncGPU(); // Make sure all GPUs are done before freeing memory.
	// Free the value arrays because they are not needed any more.
	this->closeMergeSortSmpl();
	// for (int p=0;  p<=dim; p++) { // NOTE TODO: should we free the memory ???? as we free it at ~Gpu() func
	// 	checkCudaErrors(cudaFree(this->d_values[p]));
	// }
}


/*
 * buildKdTreeGPU is a Gpu class method that prepares the data for the gpu side partitioning and build tree
 * and then calls those functions, once for each level.  Note that all of the data should already be in the GPU
 * after the mergeSortGPU function has been called.  If any of those gpu data pointers are null, this function will
 * either create them or error out.
 * Inputs
 * numTuples        number of coordinates to build the tree on.  all duplicates should be removed
 * startP			the first dimension to start the partitioning on.
 * dim              dimension of the coordinates
 */

refIdx_t Gpu::buildKdTreeGPU(const sint numTuples, const int startP, const sint dim) {
	setDevice();

	// Check to see if the GPU already has the references arrays and error out if not.
	if (d_references == NULL) {
		throw runtime_error("buildKdTree Error: device does not have the reference arrays");
	} else {
		for (sint i = 0; i < dim; i++)
			if (d_references[i] == NULL) {
				std::stringstream _str_stream;
				_str_stream << "buildKdTree Error: device does not have the reference array " << i;
				throw runtime_error(_str_stream.str());
			}
	}
	if (d_references[dim] == NULL) {  // If the last array in not there create it
		checkCudaErrors(cudaMalloc((void **) &d_references[dim], (numTuples)*sizeof(int)));
	}

	const sint tuplesDepth = int(floor(std::log2(float(numTuples))));
	for (sint i=0;  i<tuplesDepth-1; i++) {
		sint p = (i + startP) % dim;
		partitionDim( d_kdNodes, d_coord, d_references, p, dim, numTuples, i, numBlocks*numThreads);
	}
	sint p = (tuplesDepth + startP - 1) % dim;
	partitionDimLast( d_kdNodes, d_coord, d_references, p, dim, numTuples, tuplesDepth-1, numBlocks*numThreads);

	return rootNode;
}

/*
 * buildKdTree is a static Gpu class method that starts the partitioning in one or two GPUs by calling buildTreeGPU
 * Note that all of the data should already be in the GPU after the mergeSort function has been called.
 * Inputs
 * numTuples        number of coordinates to build the tree on.  all duplicates should be removed
 * dim              dimension of the coordinates
 *
 * Return 			index of the root node in the KdNodes array,.
 */
refIdx_t Gpu::buildKdTree(KdNode kdNodes[], const sint numTuples, const sint dim) 
{
	this->initBuildKdTree();
	this->buildKdTreeGPU(this->num, 0, dim);
	this->closeBuildKdTree();

	return this->rootNode;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////


/*
 * cuVerifyKdTree is a GPU kernel that is used to verify that one depth level the kdTree is correct. The refs array hold the indices of the
 * KdNodes at a particular level of the tree. For each entry in refs, this kernel will examine the child nodes to make sure that
 * the coordinate of the ltChild is less than coordinate of self and the coordinate of the gtChild is greater than itself.  If this test
 * fails, the negated index of self is written to the nextRefs array.  In addition the global d_verifyKTdreeError be written with an
 * error code so that the cpu function only has to check the one variable for pass or fail.
 * At the first call to this kernel for level 0, refs contain a single index which is the root of the tree. If no error is found, the indices
 * of the ltChild and gtChild are written to the nextRefs array.  The nextRefs array will be used as the refs array on the next call to this
 * kernel. The nextRefs array must be double the size of the refs array.
 * Finally each block of threads of the kernel adds the number of good knNoes it found to the g_sum array.
 * Inputs
 * kdNodes[]       Pointer to the KdNodes array
 * coord[]         Pointer to the coordinates array
 * refs[]          pointer to the array of indices of the KdNodes to be tested.
 * num             sint indicating the number of indices in the refs array to be tested
 * p               sint indicating the primary coordinate for this level
 * dim             sint indicating the number fo dimensions
 * Outputs
 * nextRefs        Pointer to the array where the indices of the child nodes will be stored
 * g_sum           pointer to the array where the count of good nodes found by each block will be held
 */

// TODO use a negative value in g_sums[0] to indicate an error instead so a global is not needed.
// __device__ uint d_verifyKdTreeError;

__global__ void cuAssignment(KdNode kdNodes[], refIdx_t nextRefs[], refIdx_t refs[], const sint num, const sint p)
{
	const sint pos = threadIdx.x + blockIdx.x * blockDim.x;
	const sint numThreads = gridDim.x * blockDim.x;

	refIdx_t node;
	refIdx_t parent;
	if (num == 1)
	{
		node = refs[0];
		kdNodes[node].split_dim = p;
		kdNodes[node].parent    = -1;
		kdNodes[node].brother   = -1;
	}
	else
	{
		for (sint i = pos;  i<num; i+=numThreads) // assign values for kdNodes
		{
			node = refs[i];
			if (node > -1) 
			{
				KdNode& node_current = kdNodes[node];

				node_current.split_dim = p;

				parent = nextRefs[i / 2]; // read from previous buffer
				node_current.parent = parent;
	
				node_current.brother = (kdNodes[parent].ltChild == node) ? (kdNodes[parent].gtChild)
																		 : (kdNodes[parent].ltChild);
				
			}
		}
	}
}

// TODO create a __device__ function to handle the summation within a block.
__global__ void cuVerifyKdTree(const KdNode kdNodes[], const KdCoord coord[], sint g_sums[], refIdx_t nextRefs[], refIdx_t refs[], const sint num, const sint p, const sint dim, uint* d_verifyKdTreeError) {

	const sint pos = threadIdx.x + blockIdx.x * blockDim.x;
	const sint tid = threadIdx.x;
	const sint numThreads = gridDim.x * blockDim.x;

	__shared__ sint s_sums[SHARED_SIZE_LIMIT];
	sint myCount = 0;

	refIdx_t node;
	for (sint i = pos;  i<num; i+=numThreads) {
		node = refs[i];
		if (node > -1) { // Is there a node here?
			myCount++; // Count the node.
			refIdx_t child = kdNodes[node].gtChild; // Save off the gt node
			nextRefs[i*2+1] = child; // Put the child in the refs array for the next loop
			if (child != -1) { // Check for proper comparison
				KdCoord cmp = cuSuperKeyCompare(coord+kdNodes[child].tuple*dim, coord+kdNodes[node].tuple*dim, p, dim);
				if (cmp <= 0) {  // gtChild .le. self is an error so indicate that.
					//	  nextRefs[i*2+1] = -node;  // Overwrite the child with the error code
					*d_verifyKdTreeError = 1; // and mark the error
				}
			}

			// now the less than side.
			child = kdNodes[node].ltChild;
			nextRefs[i*2] = child; // Put the child in the refs array for the next loop
			if (child != -1) {
				KdCoord cmp = cuSuperKeyCompare(coord+kdNodes[child].tuple*dim, coord+kdNodes[node].tuple*dim, p, dim);
				if (cmp >= 0) {  // gtChild .ge. self is an error so indicate that.
					//	  nextRefs[i*2] = -node;  // Overwrite the child with the error code
					*d_verifyKdTreeError = 1;
				}
			}
		} else {
			nextRefs[i*2]   = -1;  // If there was no nod here, make sure the next level knows that
			nextRefs[i*2+1] = -1;
		}
	}
	s_sums[tid] = myCount;
	// Now sum up the number of nodes found using the standard Cuda reduction code.
	__syncthreads();

	for (sint s=blockDim.x/2; s>32; s>>=1)
	{
		if (tid < s)
			s_sums[tid] = myCount = myCount + s_sums[tid + s];
		__syncthreads();
	}
	if (tid<32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockDim.x >=  64) myCount += s_sums[tid + 32];
		// Reduce final warp using shuffle
		for (sint offset = warpSize/2; offset > 0; offset /= 2)
		{
			myCount += __shfl_down(myCount, offset);
		}
	}
	if (tid == 0)
		g_sums[blockIdx.x] += myCount; // and save off this block's sum.
}

/*
 * blockReduce is a kernel which adds sums all of the values in the g_sum array.  The final
 * sum is returned in g_sum[0].
 * Inputs:
 * g_sum[]      pointer to the array to be summed
 * N            sint indicating number of values to be summed
 * Output
 * g_sum[0]     contains the final sum.
 */

__global__ void blockReduce(sint g_sums[], sint N) {

	const sint numThreads = gridDim.x * blockDim.x;
	const sint tid = threadIdx.x;

	__shared__ sint s_sums[SHARED_SIZE_LIMIT];
	sint mySum = 0;

	// Read in the data to be summed
	for (sint i = tid;  (i)<N; i+=numThreads) {
		mySum += g_sums[i];
	}
	s_sums[tid] = mySum;
	// Now sum up the number of nodes found using the standard Cuda reduction code.
	__syncthreads();
	for (uint s=blockDim.x/2; s>32; s>>=1)
	{
		if (tid < s)
			s_sums[tid] = mySum = mySum + s_sums[tid + s];
		__syncthreads();
	}
	if (tid < 32)
	{
		// Fetch final intermediate sum from 2nd warp
		if (blockDim.x >=  64) mySum += s_sums[tid + 32];
		// Reduce final warp using shuffle
		for (sint offset = warpSize/2; offset > 0; offset /= 2)
		{
			mySum += __shfl_down(mySum, offset);
		}
	}
	if (tid == 0)
		g_sums[blockIdx.x] = mySum; // Save off this blocks sum.
}

/*
 * verifyKdTreeGPU is a Gpu class method that sets up the data for the cuVerifyKdTree kernel and
 * calls it once for each level of the kdTree.  After each call it checks to see if the GPU kernel
 */
void Gpu::initVerifyKdTree() {
#pragma omp critical (launchLock)
	{
		setDevice();
		// Allocate the arrays to store the midpoint references for this level
		checkCudaErrors(cudaMalloc((void **)&d_midRefs[0], 2 * num * sizeof(refIdx_t)));
		checkCudaErrors(cudaMalloc((void **)&d_midRefs[1], 2 * num * sizeof(refIdx_t)));
		// Allocate and 0 out the partial sum array used to count the number of nodes.
		checkCudaErrors(cudaMalloc((void **)&d_sums, numBlocks * sizeof(uint)));
		checkCudaErrors(cudaMemset(d_sums, 0, numBlocks * sizeof(uint)));
	}
}
void Gpu::closeVerifyKdTree() {
	syncGPU();
#pragma omp critical (launchLock)
	{
		setDevice();
		// Free the arrays to store the midpoint references for this level
		checkCudaErrors(cudaFree(d_midRefs[0]));
		checkCudaErrors(cudaFree(d_midRefs[1]));
		checkCudaErrors(cudaFree(d_sums));
	}
}

/*
 * verifyKdTreeGPU is a Gpu class method that sets up the data for the cuVerifyKdTree kernel and
 * calls it once for each level of the kdTree.  After each call it checks to see if the GPU kernel
 * found an error and if so, some of the errors are printed and then the program exits.
 * Inputs
 * root            index of the root node in the kdNodes array
 * startP          axis to start on.  should always be less than dim
 * dim             number of dimensions
 * numTuples       number of KdNodes
 *
 * Return          number of kdNodes found
 */
sint Gpu::verifyKdTreeGPU(const sint root, const sint startP, const sint dim, const sint numTuples) {
	setDevice();
	const sint logNumTuples = int(floor(std::log2(float(numTuples))));

	// Put the root node in the children array for level 0
	checkCudaErrors(cudaMemcpyAsync(d_midRefs[0], &root, sizeof(refIdx_t), cudaMemcpyHostToDevice, stream));

	refIdx_t* nextChildren; // Used to setGPU the pointer to the where children will be put
	refIdx_t* children;     // Used to setGPU the pointer where the children will be read

	// Clear the error flag in the GPU
	sint verifyKdTreeError = 0;
	// cudaMemcpyToSymbolAsync(d_verifyKdTreeError, &verifyKdTreeError,
	// 		sizeof(verifyKdTreeError),
	// 		0,cudaMemcpyHostToDevice, stream);
	checkCudaErrors(cudaMemcpyAsync(d_verifyKdTreeError, &verifyKdTreeError, sizeof(uint), cudaMemcpyHostToDevice, stream));

	// Loop through the levels
	for (sint level = 0;  level < logNumTuples+1; level++) {

		const sint p = (level+startP) % dim; // Calculate the primary axis for this level
		nextChildren = d_midRefs[(level+1) % 2];
		children = d_midRefs[(level) % 2];
		// Allocate the array to put the children of this level in. Needs to be twice the size of the current level.

		// Check the current level and get the nodes for the next level.  Only start enough thread to cover current level
		sint threadsNeeded = 1<<level;
		sint blocks;
		// Calculate the right thread and block numbers
		if (threadsNeeded > numThreads){
			blocks = threadsNeeded/numThreads;
			if (blocks > numBlocks) blocks = numBlocks;
		} else {
			blocks = 1;
		}
#pragma omp critical (launchLock)
		{
			setDevice();
			cuAssignment<<<blocks,numThreads, 0, stream>>>(d_kdNodes,
				nextChildren,
				children, 
				(1<<level), 
				p);
			checkCudaErrors(cudaGetLastError());
			cuVerifyKdTree<<<blocks,numThreads, 0, stream>>>(d_kdNodes,
					d_coord,
					d_sums,
					nextChildren,
					children,
					(1<<level), p, dim,
					d_verifyKdTreeError);
			checkCudaErrors(cudaGetLastError());
		}
		// Check for error on the last run
		// cudaMemcpyFromSymbolAsync(&verifyKdTreeError,
		// 		d_verifyKdTreeError,
		// 		sizeof(verifyKdTreeError),
		// 		0,cudaMemcpyDeviceToHost, stream);
		checkCudaErrors(cudaMemcpyAsync(&verifyKdTreeError, d_verifyKdTreeError, sizeof(uint), cudaMemcpyDeviceToHost, stream));
		syncGPU();  // Wait
		if (verifyKdTreeError != 0){  // See if the kernel for this level found an error
			cout << "Verify Tree Error at level " << level << endl;
			// Here is where we get the data back from the GPU and find the node with the arror
			refIdx_t* h_children = new refIdx_t[2<<level];
			checkCudaErrors(cudaMemcpyAsync(h_children, nextChildren, (2<<level)*sizeof(refIdx_t), cudaMemcpyDeviceToHost, stream));
			cout << "First 10 nodes in error are ";
			sint cnt = 0;
			sint all = 2 << level;
			for (sint i = 0; i < all; i++){
				if (h_children[i] < 0) // Is it a failure?
					if (cnt++ < 10) // Only print the first ten failures.
						cout << "[" << i << "]" << -h_children[i] << " ";
			}
			cout << endl << "Total of " << cnt << " bad nodes found" << "out of " << all << endl;
			return -1;
		}
	}

	// Finally, add the sums of all the blocks together and return the final count.
#pragma omp critical (launchLock)
	{
		setDevice();
		blockReduce<<<1,numThreads, 0, stream>>>(d_sums, numBlocks);
		checkCudaErrors(cudaGetLastError());
	}
	sint numNodes;
	checkCudaErrors(cudaMemcpyAsync(&numNodes, d_sums, sizeof(numNodes), cudaMemcpyDeviceToHost, stream));

	return numNodes;
}

/*
 * verifyKdTree is a static Gpu class method that set up and calls the verifyKdTreeGPU method
 * on each GPU.
 * Inputs
 * kdNodes[]       Pointer the the cpu copy of the kdNodes array.  Currently unused
 * root            index of the root node in the kdNodes array
 * dim             number of dimensions
 * numTuples       number of KdNodes
 *
 * Return          Total number of kdNodes found
 */
sint Gpu::verifyKdTree(KdNode kdNodes[], const sint root, const sint dim, const sint numTuples) {
	// Set up memory for the verify tree functions
	this->initVerifyKdTree();

	int nodeCnt = this->verifyKdTreeGPU(root, 0, dim, numTuples);

	// free the memory used for verifying the tree.
	this->closeVerifyKdTree();

	return nodeCnt;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/*
 * getKdNodesFromGPU is a Gpu class method that copies the GPU version of the KdNodes array to the host.
 * Inputs
 * numTuples         size of kdNodes array to copy
 * Output
 * kdNodes[]         pointer to the place where the kdNodes will be copied
 */
void Gpu::getKdNodesFromGPU(KdNode kdNodes[], const sint numTuples){
	setDevice();
	// Copy the knNodes array back
	if (kdNodes != NULL && d_kdNodes != NULL){
		checkCudaErrors(cudaMemcpyAsync(kdNodes, d_kdNodes, (numTuples)*sizeof(KdNode), cudaMemcpyDeviceToHost, stream));
	} else {
		if (kdNodes == NULL)
			throw runtime_error("getKdNodesFromGPU Error: Don't know where to put the kdNodes");
		if (d_kdNodes == NULL)
			throw runtime_error("getKdNodesFromGPU Error: GPU copy of kdNodes is not available");
	}
}

/*
 * getKdTreeResults is a static Gpu class method that copies the KdNoes data and coordinate data
 * from all GPUs to a local copy.  Data from each GPU is concatenated into a single array so in
 * the two GPU case, the returned indices need to be fixed.
 * Inputs
 * numTuples         size of kdNodes array to copy
 * Output
 * kdNodes[]        Host KdNodes array where data from the GPU should be put
 * coord[]          Host coordinate array where data from the GPU should be put
 *                    * This is only used for the 2 GPU case where coordinate data
 *                       may get reordered
 */
void Gpu::getKdTreeResults(KdNode kdNodes[], const sint numTuples) {
	// Copy the knNodes array back
	if (kdNodes != NULL ){
		this->getKdNodesFromGPU(kdNodes, numTuples);
	} else {
		throw runtime_error("getKdTreeResults Error: Don't know where to put the kdNodes");
	}
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

/*
 * gpuSetup is a static class method that determines what GPUs are available and whether or
 * not they are capable of UVM which is required for multi-GPU sort.  It then creates an
 * instance of the GPU class for each GPU to be used.
 * Inputs
 * gpu_max			Maximum nmber of GPUs to apply.  (cannot be greater than 2 for this version)
 * threads			Maximum number of threads to use on partitioning
 * blocks			Macimum number of blocks to use on partitioning
 */
Gpu* Gpu::gpuSetup(int threads, int blocks, int dim, int gpuid, cudaStream_t torch_stream)
{
	// NOTE: gpuid is the index of the used visible gpu device
	Gpu* gpu_ptr = new Gpu(threads, blocks, dim, gpuid, torch_stream);
	if (gpu_ptr == nullptr)
	{
		throw runtime_error("gpuSetup Error: Fail to allocate cuda device");
	}
	return gpu_ptr;
}



/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

__global__ void cuInitSearch(sint num_of_points, 
                             CoordStartEndIndices* d_index_down,  refIdx_t root_index, 
                             sint* d_stack_back)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_of_points)
    {
		d_index_down[tid].coord_index = tid;
		d_index_down[tid].start_index = root_index;
		d_index_down[tid].end_index   = root_index;
        d_stack_back[tid] = -1;
    }
}

void Gpu::InitQueryMem(sint _num_of_points, SearchType search)
{	
	sint max_allocated_npoints, max_allocated_result_size;
	std::tie(max_allocated_npoints, max_allocated_result_size) = max_allocated_size;

	if (_num_of_points > max_allocated_npoints) // memory not enough
	{
		DestroyQueryMem();
		checkCudaErrors(cudaMalloc((void**)&d_index_temp, sizeof(CoordStartEndIndices) * _num_of_points));
		checkCudaErrors(cudaMalloc((void**)&d_index_down, sizeof(CoordStartEndIndices) * _num_of_points));
		checkCudaErrors(cudaMalloc((void**)&d_index_up, sizeof(CoordStartEndIndices) * _num_of_points));
		checkCudaErrors(cudaMalloc((void**)&d_num_temp, sizeof(sint)));
		checkCudaErrors(cudaMalloc((void**)&d_num_down, sizeof(sint)));
		checkCudaErrors(cudaMalloc((void**)&d_num_up, sizeof(sint)));
		checkCudaErrors(cudaMalloc((void**)&d_stack, sizeof(StartEndIndices) * _num_of_points * CUDA_STACK_MAX));
		checkCudaErrors(cudaMalloc((void**)&d_stack_back, sizeof(sint) * _num_of_points));
		checkCudaErrors(cudaMalloc((void**)&d_num_empty, sizeof(sint)));
	}

	sint requires_allocated_size = 0;
	if (search == Nearest) requires_allocated_size = sizeof(ResultNearest) * _num_of_points;
	else if (search == Knn) throw runtime_error("not implemented");
	else if (search == Radius) throw runtime_error("not implemented");
	if (requires_allocated_size > max_allocated_result_size)
	{
		if (d_result_buffer != nullptr) checkCudaErrors(cudaFree(d_result_buffer));
		checkCudaErrors(cudaMalloc((void**)&d_result_buffer, requires_allocated_size));
	}
	
	max_allocated_npoints = std::max(max_allocated_npoints, _num_of_points);
	max_allocated_result_size = std::max(max_allocated_result_size, requires_allocated_size);
	max_allocated_size = std::make_tuple(max_allocated_npoints, max_allocated_result_size);

	num_of_points = _num_of_points; // num of querying points
}

void Gpu::DestroyQueryMem()
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
	if (d_stack != nullptr)
		checkCudaErrors(cudaFree(d_stack));
	if (d_stack_back != nullptr)
		checkCudaErrors(cudaFree(d_stack_back));
	if (d_num_empty != nullptr)
		checkCudaErrors(cudaFree(d_num_empty));
}


void Gpu::InitSearch(sint _num_of_points, SearchType search)
{
	InitQueryMem(_num_of_points, search);

	sint zero_sint = 0;
	sint num_sint = num_of_points;
	checkCudaErrors(cudaMemcpyAsync(d_num_temp, &zero_sint, sizeof(sint), cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_num_down, &num_sint, sizeof(sint), cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_num_up, &zero_sint, sizeof(sint), cudaMemcpyHostToDevice, stream));

	const int total_num = num_of_points;
	const int thread_num = std::min(numThreads, total_num);
	const int block_num = int(std::ceil(total_num / float(thread_num)));
	cuInitSearch<<<block_num, thread_num, 0, stream>>>(num_of_points, d_index_down, rootNode, d_stack_back);
	checkCudaErrors(cudaGetLastError());
}

