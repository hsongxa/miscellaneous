#include "cuda_kernels.cuh"

__global__ void kernel_axpby(double a, const double* x, int n, double b, double* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b * y[i];
}