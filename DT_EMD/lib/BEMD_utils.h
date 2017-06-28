#pragma once
#include <cstddef>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <stdio.h>

#include "math.h"


//-----------------------------------------------------
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//-----------------------------------------------------

#define EXTREMA_SIZE 40*BUFSIZ
#define MAX 0
#define MIN 1



__host__ void check(const char* message);

__device__ __host__ int CeilDiv(int a, int b);
__device__ __host__ int CeilAlign(int a, int b);

__global__ void set_2D(float** array_2D, float* array_1D, const int width, const int height);
__global__ void deep_copy(float** target, float** source, const int width, const int height);
void minus_value(float* data, float value, int size);
void minus_data(float* data, const float* value, int size);

__host__ __device__ void array_add(float* input, float value, int size);


