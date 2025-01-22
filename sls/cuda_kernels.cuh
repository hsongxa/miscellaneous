#ifndef CONJUGATE_GRADIENT_H
#define CONJUGATE_GRADIENT_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void kernel_axpby(double a, const double* x, int n, double b, double* y);


#endif
