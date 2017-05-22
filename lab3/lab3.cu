#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__device__ void RGB_add(float* RGB, const float* value) {
	for (int i = 0; i < 3; i++) {
		RGB[i] += value[i];
	}
}

__device__ void RGB_sub(float* RGB, const float* value) {
	for (int i = 0; i < 3; i++) {
		RGB[i] -= value[i];
	}
}

__global__ void checkBorder(
	float* mask_border,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt + xt;
	if (0 > oy + yt) {mask_border[curt] = 0;}
	if (hb <= oy + yt) {mask_border[curt] = 0;}
	if (0 > ox + xt) { mask_border[curt] = 0; }
	if (wb <= ox + xt) {mask_border[curt] = 0;}
}

__global__ void CalculateFixed(
	const float* background,
	const float* target,
	const float* mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt + xt;
	const int glob = wb*(oy+yt)+(ox+xt) ;
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		float RGB[3] = {0, 0, 0};
		int neighbors = 0;
		// upper
		if (yt != 0) {
			RGB_sub(RGB, &target[(curt - wt) * 3]);
			neighbors++;
			if (mask[curt - wt] < 127) {
				if (0 >= oy + yt) { RGB_add(RGB, &background[(glob) * 3]); }
				else { RGB_add(RGB, &background[(glob - wb) * 3]); }
			}
		}
		else {
			if (0 >= oy + yt) { RGB_add(RGB, &background[(glob) * 3]); }
			else { RGB_add(RGB, &background[(glob - wb) * 3]); }
		}
		// lower
		if (yt != ht-1) {
			RGB_sub(RGB, &target[(curt + wt) * 3]);
			neighbors++;
			if (mask[curt + wt] < 127) {
				if (hb <= oy + yt + 1) { RGB_add(RGB, &background[(glob) * 3]); }
				else { RGB_add(RGB, &background[(glob + wb) * 3]); }
			}
		}
		else {
			if (hb <= oy + yt + 1) { RGB_add(RGB, &background[(glob) * 3]); }
			else { RGB_add(RGB, &background[(glob + wb) * 3]); }
		}
		// left
		if (xt != 0) {
			RGB_sub(RGB, &target[(curt - 1) * 3]);
			neighbors++;
			if (mask[curt - 1] < 127) {
				if (0 >= ox + xt - 1) { RGB_add(RGB, &background[(glob) * 3]); }
				else { RGB_add(RGB, &background[(glob - 1) * 3]); }
			}
		}
		else {
			if (0 >= ox + xt) { RGB_add(RGB, &background[(glob) * 3]); }
			else { RGB_add(RGB, &background[(glob - 1) * 3]); }
		}
		// right
		if (xt != wt - 1) {
			RGB_sub(RGB, &target[(curt + 1) * 3]);
			neighbors++;
			if (mask[curt + 1] < 127) {
				if (wb <= ox + xt + 1) {RGB_add(RGB, &background[(glob) * 3]);}
				else { RGB_add(RGB, &background[(glob + 1) * 3]); }
			}
		}
		else {
			if(wb <= ox + xt + 1){ RGB_add(RGB, &background[(glob) * 3]); }
			else { RGB_add(RGB, &background[(glob + 1) * 3]); }
		}
		
		RGB[0] += neighbors * target[curt * 3];
		RGB[1] += neighbors * target[curt * 3 + 1];
		RGB[2] += neighbors * target[curt * 3 + 2];
		
		fixed[curt * 3] = RGB[0]/4;
		fixed[curt * 3 + 1] = RGB[1]/4;
		fixed[curt * 3 + 2] = RGB[2]/4;
	}
}

__global__ void PoissonImageCloningIteration(
	const float* fixed, 
	const float* mask, 
	float* buf1, float* buf2, 
	const int wt, const int ht
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt + xt;
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		float RGB[3] = { 0, 0, 0 };
		if (yt != 0 && mask[curt - wt] > 127.0f) {RGB_add(RGB, &buf1[(curt - wt) * 3]);}
		if (yt != ht-1 && mask[curt + wt] > 127.0f) { RGB_add(RGB, &buf1[(curt + wt) * 3]);} 
		if (xt != 0 && mask[curt - 1] > 127.0f) { RGB_add(RGB, &buf1[(curt - 1) * 3]);} 
		if (xt != wt-1 && mask[curt + 1] > 127.0f) { RGB_add(RGB, &buf1[(curt + 1) * 3]);} 

		buf2[curt * 3] = fixed[curt * 3] + RGB[0] / 4;
		buf2[curt * 3 + 1] = fixed[curt * 3 + 1] + RGB[1] / 4;
		buf2[curt * 3 + 2] = fixed[curt * 3 + 2] + RGB[2] / 4;
	}

}

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

void PoissonImageCloning(const float *background, const float *target, const float *mask, float *output,
	const int wb, const int hb, const int wt, const int ht, const int oy, const int ox)
{
	// set up
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3 * wt*ht * sizeof(float));
	cudaMalloc(&buf1, 3 * wt*ht * sizeof(float));
	cudaMalloc(&buf2, 3 * wt*ht * sizeof(float));
	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);

	// check target exceed border
	float *mask_border;
	cudaMalloc(&mask_border, wt*ht * sizeof(float));
	cudaMemcpy(mask_border, mask, sizeof(float)*wt*ht, cudaMemcpyDeviceToDevice);
	checkBorder <<<gdim, bdim>>>(mask_border, wb, hb, wt, ht, oy, ox);

	// initialize the iteration
	
	printf("initializing\n");

	CalculateFixed <<<gdim, bdim>>> (background, target, mask_border, fixed, wb, hb, wt, ht, oy, ox);
	cudaMemcpy(buf1, target, sizeof(float) * 3 * wt*ht, cudaMemcpyDeviceToDevice);
	
	
	for (int i = 0; i < 10000; ++i) {
		printf("iteration: %d\n", i);
		PoissonImageCloningIteration <<<gdim, bdim>>> (fixed, mask_border, buf1, buf2, wt, ht);
		PoissonImageCloningIteration <<<gdim, bdim>>> (fixed, mask_border, buf2, buf1, wt, ht);
	}
	
	cudaMemcpy(output, background, wb * hb * sizeof(float) * 3, cudaMemcpyDeviceToDevice);
	SimpleClone <<<gdim, bdim >>>( background, buf1, mask_border, output,
		wb, hb, wt, ht, oy, ox);

	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
	cudaFree(mask_border);
}
