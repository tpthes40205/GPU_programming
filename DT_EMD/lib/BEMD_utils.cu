#include "BEMD_utils.h"


__host__ void check(const char* message) {
	auto e = cudaDeviceSynchronize();
	if (e != cudaSuccess) {
		printf("%s\n", message);
		abort();
	}
}

__device__ __host__ int CeilDiv(int a, int b) { return (a - 1) / b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void set_2D(float** array_2D, float* array_1D, const int width, const int height) {
	for (int i = 0; i < height; i++) {
		array_2D[i] = &array_1D[i*width];
	}
}



__global__ void deep_copy(float** target, float** source, const int width, const int height) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			target[i][j] = source[i][j];
		}
	}
}

void minus_value(float* data, float value, int size) {
	for (int i = 0; i < size; i++) { data[i] -= value; }
}
void minus_data(float* data, const float* value, int size) {
	for (int i = 0; i < size; i++) { data[i] -= value[i]; }
}


__host__ __device__ void array_add(float* input, float value, int size) {
	for (int i = 0; i < size; i++) {
		input[i] += value;
	}
}

