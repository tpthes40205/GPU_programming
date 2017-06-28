#include "BEMD.h"


__global__ void set_map_to(float** map, const int width, const int height, const int value) {
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			map[j][i] = value;
		}
	}
}

__global__ void draw_corner(float** output, int* count,
	int* extrema_x, int* extrema_y, float* extrema_value,
	const int width, const int height, int mode) {
	count[mode] = 0;
	int id = 0;
	for (int j = 0; j < height; j += (height - 1)) {
		for (int i = 0; i < width; i += (width - 1)) {
			extrema_y[id] = j;
			extrema_x[id] = i;
			extrema_value[id] = output[j][i];
			id++;
			count[mode]++;
		}
	}
}

__global__ void draw_extrema(float** map, int* count, int mode, int* extrema_x, int* extrema_y, float* extrema_value) {
	int i, pt_count = count[mode];
	for (i = 0; i < pt_count; i++) {
		//printf("(%d, %d): %d\n", extrema_y[i], extrema_x[i], extrema_value[i]);
		map[extrema_y[i]][extrema_x[i]] = extrema_value[i];
	}
}

__global__ void find_max(float** output, int* count,
	int* extrema_x, int* extrema_y, float* extrema_value,
	const int width, const int height){
	const int SIZE = height * width;
	float value;
	int max_count, neighbor_count, equal_count ,id;
	int i, j;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			neighbor_count = 4;
			max_count = 0;
			equal_count = 0;
			

			value = output[i][j];
			if (i == 0) { neighbor_count--; }
			else {if (value >= output[i - 1][j]) {
				if (value == output[i - 1][j]) { equal_count++; }
				max_count++;}
			}
			
			if (i == height - 1) { neighbor_count--; }
			else {if (value >= output[i + 1][j]) { 
				if (value == output[i + 1][j]) { equal_count++; }
				max_count++; }
			}
			
			if (j == 0) { neighbor_count--; }
			else {if (value >= output[i][j - 1]) { 
				if (value == output[i][j - 1]) { equal_count++; }
				max_count++; }
			}
			
			if (j == width - 1) { neighbor_count--; }
			else {if (value >= output[i][j + 1]) { 
				if (value == output[i][j + 1]) { equal_count++; }
				max_count++; }
			}
			
			if (max_count == neighbor_count && max_count != equal_count && neighbor_count != 2) {
				id = count[MAX];
				extrema_y[id] = i;
				extrema_x[id] = j;
				extrema_value[id] = value;
				count[MAX]++;
			}
		}
	}
	if (count[MAX] > EXTREMA_SIZE) { asm("trap;"); }
}

__global__ void find_min(float** output, int* count,
	int* extrema_x, int* extrema_y, float* extrema_value,
	const int width, const int height) {
	const int SIZE = height * width;
	float value;
	int min_count, neighbor_count, equal_count, id;
	int i, j;
	
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			neighbor_count = 4;
			min_count = 0;
			equal_count = 0;


			value = output[i][j];
			if (i == 0) { neighbor_count--; }
			else {
				if (value <= output[i - 1][j]) {
					if (value == output[i - 1][j]) { equal_count++; }
					min_count++;
				}
			}

			if (i == height - 1) { neighbor_count--; }
			else {
				if (value <= output[i + 1][j]) {
					if (value == output[i + 1][j]) { equal_count++; }
					min_count++;
				}
			}

			if (j == 0) { neighbor_count--; }
			else {
				if (value <= output[i][j - 1]) {
					if (value == output[i][j - 1]) { equal_count++; }
					min_count++;
				}
			}

			if (j == width - 1) { neighbor_count--; }
			else {
				if (value <= output[i][j + 1]) {
					if (value == output[i][j + 1]) { equal_count++; }
					min_count++;
				}
			}

			if (min_count == neighbor_count && min_count != equal_count && neighbor_count != 2) {
				id = count[MIN];
				extrema_y[id] = i;
				extrema_x[id] = j;
				extrema_value[id] = value;
				count[MIN]++;
			
			}
		}
	}
	if (count[MIN] > EXTREMA_SIZE) { asm("trap;"); }
}

__global__ void minus_map(float** layer, float** max_map, float** min_map, const int width, const int height, float* gpu_residual) {
	float std, mean, value;
	mean = std = gpu_residual[0] = 0.0;

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			value = (max_map[j][i] + min_map[j][i])/2;
			mean += value;
			std += value * value;
			layer[j][i] -= value;
		}
	}
	mean = mean/ (height * width);
	std = std / (height * width) - mean * mean;
	gpu_residual[0] = sqrtf(std);
}


float BEMD(float** output, float** max_map, float** min_map, int* extrema_array,
	const int width, const int height){
	dim3 gdim(CeilDiv(width, 32), CeilDiv(height, 16)), bdim(32, 16);
	int* extrema_x[2] = { &extrema_array[0 * EXTREMA_SIZE], &extrema_array[1 * EXTREMA_SIZE] };
	int* extrema_y[2] = { &extrema_array[2 * EXTREMA_SIZE], &extrema_array[3 * EXTREMA_SIZE] };
	float* extrema_value[2] = { (float*)&extrema_array[4 * EXTREMA_SIZE], (float*)&extrema_array[5 * EXTREMA_SIZE] };
	const int SIZE = width * height;

	
	float* gpu_residual;
	float cpu_residual = 100;
	int* count;
	cudaMalloc(&gpu_residual, sizeof(float));
	cudaMalloc(&count, 2 * sizeof(int));
	

	for (int i = 0; cpu_residual > 2; i++) {
		draw_corner <<<1, 1>>>(output, count, extrema_x[MAX], extrema_y[MAX], extrema_value[MAX], width, height, MAX);
		find_max <<<1, 1 >>> (output, count, extrema_x[MAX], extrema_y[MAX], extrema_value[MAX], width, height);
		check("max count out of EXTREMA_SIZE");
		Delaunay_triangle(max_map, count, MAX, extrema_x[MAX], extrema_y[MAX], extrema_value[MAX], width, height);
		check("DT Fail");

		draw_corner <<<1, 1>>> (output, count, extrema_x[MIN], extrema_y[MIN], extrema_value[MIN], width, height, MIN);
		find_min <<<1, 1 >>> (output, count, extrema_x[MIN], extrema_y[MIN], extrema_value[MIN], width, height);
		check("min count out of EXTREMA_SIZE");
		Delaunay_triangle(min_map, count, MIN, extrema_x[MIN], extrema_y[MIN], extrema_value[MIN], width, height);
		check("DT Fail");

		minus_map<<<1, 1 >>>(output, max_map, min_map, width, height, gpu_residual);
		cudaMemcpy(&cpu_residual, gpu_residual, sizeof(float), cudaMemcpyDeviceToHost);
		printf("%f\n", cpu_residual);
	}

	//set_map_to<<<1,1>>>(output, width, height, 0);
	//draw_extrema<<<1,1>>>(output, count, MAX, extrema_x[MAX], extrema_y[MAX], extrema_value[MAX]);
	
	
	//deep_copy<<<1,1>>>(output, max_map, width, height);

	cudaFree(gpu_residual);
	cudaFree(count);
	return cpu_residual;
}