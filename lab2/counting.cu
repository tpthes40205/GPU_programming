#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <thrust/device_vector.h>

#define THREAD_SIZE 500

struct not_space
{
	__host__ __device__
		bool operator()(const char c)
	{
		return c != '\n';
	}
};

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

void CountPosition1(const char *text, int *pos, int text_size)
{
	thrust::device_ptr<const char> text_ptr(text);
	thrust::device_ptr<int> pos_ptr(pos);
	thrust::fill(pos_ptr, pos_ptr + text_size, 1);
	thrust::device_vector<int> key(text_size, 0);
	
	not_space n_space;																					// text:	asd fghj   kl
	thrust::replace_if(thrust::device,key.begin(), key.end(), text_ptr, n_space, 1);					//  key:	1110111100011
	thrust::inclusive_scan_by_key(thrust::device,key.begin(), key.end(), pos_ptr, pos_ptr);				//  pos:	1231123412312
	thrust::transform(pos_ptr, pos_ptr + text_size, key.begin(), pos_ptr, thrust::multiplies<int>());	//  pos:	1230123400012	
}

void CountPosition2(const char *text, int *pos, int text_size)
{
	int block_size = text_size/THREAD_SIZE/10000;
	count_kernel <<<block_size, THREAD_SIZE>>>(text, pos, text_size);
	check_kernel <<<block_size, THREAD_SIZE >>>(pos, text_size);
}

__global__ void count_kernel(const char *text, int *pos, int text_size) {
	
	int block_begin =((long long int)(blockIdx.x*blockDim.x + threadIdx.x)*text_size) /(gridDim.x*blockDim.x);
	int block_end = ((long long int)(blockIdx.x*blockDim.x + threadIdx.x + 1)*text_size) / (gridDim.x*blockDim.x);
	not_space n_space;

	if (n_space(text[block_begin])) {pos[block_begin] = 1;}

	for (int i = block_begin+1; i < block_end; i++) {
		if (n_space(text[i])) {pos[i] = pos[i-1] + 1;}
	}
}

__global__ void check_kernel(int *pos, int text_size) {
	if (blockIdx.x != 0 || threadIdx.x != 0) {
		int block_begin = ((long long int)(blockIdx.x*blockDim.x + threadIdx.x)*text_size) / (gridDim.x*blockDim.x);
		int i = block_begin;
		while (pos[i] != 0) {
			pos[i] = pos[i-1] + 1;
			i++;
		}
	}
}