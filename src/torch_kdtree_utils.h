
#ifndef TORCH_KDTREE_UTILS_H_
#define TORCH_KDTREE_UTILS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <string>
#include <sstream>
#include <ostream>
#include <stdexcept>
#include <stack>
#include <tuple>


// copy codes here
#include "Gpu.h"
#include "torch_kdtree_cuda_dist.h"
#include "torch_kdtree_cuda_queue.h"
#include "torch_kdtree_cuda_down.h"
#include "torch_kdtree_cuda_nearest.h"


namespace py = pybind11;

////////////////////////////////////////////////////////////////

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.dtype()==torch::kFloat32, #x " must be float32")
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define POW2(x) ((x) * (x))

////////////////////////////////////////////////////////////////

const int numThreads               = 512; // max is 1024
const int numBlocks                = 32;  // max is 1024
constexpr int numPreCompileDimsMax = 32;  // the max num of dims to precompile

////////////////////////////////////////////////////////////////

template<template<int> typename F, int N, int Nmax>
struct _DispatcherImpl
{
    template<typename... Args>
    static void dispatch(int n, Args... args)
    {
        if (n == N) F<N>::work(args...);
        else _DispatcherImpl<F, N + 1, Nmax>::dispatch(n, args...);
    }
};


template<template<int> typename F, int Nmax>
struct _DispatcherImpl<F, Nmax, Nmax>
{
    template<typename... Args>
    static void dispatch(int n, Args... args)
    {
        F<0>::work(args...);
    }
};


template<template<int> typename F, int Nmax = numPreCompileDimsMax>
struct Dispatcher
{
    template<typename... Args>
    static void dispatch(int n, Args... args)
    {
        _DispatcherImpl<F, 1, Nmax + 1>::dispatch(n, args...);
    }
};

#endif