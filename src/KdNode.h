//
//  KdNode.h
//
// This file contains the declaration of the KdNode which is the contents of
// the KdTree as well as some of the basic types to make the code more flexable.
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


#ifndef KDNODE_H_
#define KDNODE_H_

#include <stdbool.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <stdint.h>
using std::setprecision;

typedef int32_t KdCoord;
typedef int32_t refIdx_t;

typedef int32_t sint;
typedef uint32_t uint;


/* One node of a k-d tree */
class KdNode
{
public:
	int32_t tuple;              // index of the coordinate of this node.
	refIdx_t ltChild, gtChild;  // index into the kdNodes array of the gt and lt child nodes
	
	int32_t split_dim;          // the considered splitting dim
	refIdx_t parent;            // parent's index of kdNodes array
	refIdx_t brother;           // brother's index of kdNodes array

public:
	KdNode(int32_t _tuple=0, refIdx_t _ltChild=0, refIdx_t _gtChild=0): tuple(_tuple), ltChild(_ltChild), gtChild(_gtChild), 
			split_dim(0), parent(-1), brother(-1) {}

public:
	inline int32_t getTuple()
	{
		return tuple;
	}

	/* initialize kdNodes array
	 */
	static void initializeKdNodesArray(KdNode kdNodes[], KdCoord coordinates[], sint numTuples);

	/*
	 * Initialize a reference array by creating references into the coordinates in the kdNode array.
	 *
	 * calling parameters:
	 *
	 * kdNodes - a vector<KdNodes*> that contain the coordinates ie. (x, y, z, w...) tuples
	 * reference - a pointer to one of the reference arrays with indices into the kdNodes array.
	 */
private:
	static void initializeReference(refIdx_t reference[], sint numTuples);

	/*
	 * The superKeyCompare method compares two int arrays in all k dimensions,
	 * and uses the sorting or partition coordinate as the most significant dimension.
	 *
	 * calling parameters:
	 *
	 * a - a refIdx_t*
	 * b - a refIdx_t*
	 * p - the most significant dimension
	 * dim - the number of dimensions
	 *
	 * returns: +1, 0 or -1 as the result of comparing two int arrays
	 */
private:
	static KdCoord superKeyCompare(const KdCoord *a, const KdCoord *b, const sint p, const sint dim);

	/*
	 * Walk the k-d tree and check that the children of a node are in the correct branch of that node.
	 *
	 * calling parameters:
	 *
	 * dim - the number of dimensions
	 * depth - the depth in the k-d tree
	 *
	 * returns: a count of the number of kdNodes in the k-d tree
	 */
public:
	sint verifyKdTree( const KdNode kdNodes[], const KdCoord coords[], const sint dim, const sint depth) const;

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
	static KdNode *createKdTree( KdNode kdNodes[], KdCoord coordinates[],  const sint numDimensions, const sint numTuples);
	/*
	 * Search the k-d tree and find the KdNodes that lie within a cutoff distance
	 * from a query node in all k dimensions.
	 *
	 * calling parameters:
	 *
	 * query - the query point
	 * cut - the cutoff distance
	 * dim - the number of dimensions
	 * depth - the depth in the k-d tree
	 *
	 * returns: a list that contains the kdNodes that lie within the cutoff distance of the query node
	 */
public:
	list<KdNode> searchKdTree(const KdNode kdNodes[], const KdCoord coordinates[], const KdCoord* query, const KdCoord cut,
			const sint dim, const sint depth) const;

	/*
	 * Print one tuple.
	 *
	 * calling parameters:
	 *
	 * tuple - the tuple to print
	 * dim - the number of dimensions
	 */
public:
	static void printTuple(const KdCoord* tuple, const sint dim);
	/*
	 * Print the k-d tree "sideways" with the root at the ltChild.
	 *
	 * calling parameters:
	 *
	 * dim - the number of dimensions
	 * depth - the depth in the k-d tree
	 */
public:
	void printKdTree(KdNode kdNodes[], const KdCoord coords[], const sint dim, const sint depth) const;
};




#endif /* KDNODE_H_ */
