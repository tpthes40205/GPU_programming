#include <cstdio>
#include <cstdlib>
#include "SyncedMemory.h"

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

const int W = 40;
const int H = 12;

__device__ __constant__ const strln[][2] = {"  Give me the permission key,        ","                            hooman   "};
__device__ __constant__ const face[][3] = {"/\\___/\\"," o   o ","  =^=  "};

__global__ void Draw(char *frame) {
	// TODO: draw more complex things here
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y < H && x < W) {
		char c;
		if (x == W-1) {
			c = y == H-1 ? '\0' : '\n';
		}else if (y == 0 || y == H-1 || x == 0 || x == W-2){
			c = ':';
		}else if (y == 2 || y == 3){
			c = strln[x-1][y-2];
		}else if (x == 3 && y > 5){
			c = '(';
		}else if (y > 5 && (x-y)==5){
			c = ')';
		}else if (y > 4 && y < 8 && x > 2 && x < 11){
			c = face[x-3][y-5];
		}else if (y==10 && x > 15 && x < 26){
			c = ')';
		}else{
			c = ' ';
		}
		frame[y*W+x] = c;
	}
}



int main(int argc, char **argv)
{
	MemoryBuffer<char> frame(W*H);
	auto frame_smem = frame.CreateSync(W*H);
	CHECK;

	Draw<<<dim3((W-1)/16+1,(H-1)/12+1), dim3(16,12)>>>(frame_smem.get_gpu_wo());
	CHECK;

	puts(frame_smem.get_cpu_ro());
	CHECK;
	return 0;
}