///////////////////////////////////////
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
///////////////////////////////////////
// remember put this back to main
#include <cstdio>
#include <cstdint>
#include <cstdlib>
///////////////////////////////////////

#include "lab1.h"

struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	info.fps_n = 30;
	info.fps_d = 1;
};

void Lab1VideoGenerator::Generate(SyncedMemory<uint8_t>* grids0, SyncedMemory<uint8_t>* grids1, SyncedMemory<uint8_t>* frames, uint8_t** gpu2DPtr0, uint8_t** gpu2DPtr1) {
	uint8_t **now2DPtr, **next2DPtr;
	int frameT = impl->t;
	
	if (frameT ==0) {
		initGrids(grids0, frames);
		initGPU<<<1,1>>>(grids0->get_gpu_rw(), gpu2DPtr0);
		initGPU<<<1,1>>>(grids1->get_gpu_rw(), gpu2DPtr1);
	}
	
	int startStep = TOTAL_STEP * frameT/NFRAME;
	int endStep = TOTAL_STEP * (frameT+1) / NFRAME;
	int step = startStep;


	if (step % 2 == 0) {
		now2DPtr = gpu2DPtr0;
		next2DPtr = gpu2DPtr1;
	}
	else {
		now2DPtr = gpu2DPtr1;
		next2DPtr = gpu2DPtr0;
	}

	auto yuv = frames->get_gpu_rw();

	Draw <<<BLOCK_SIZE, THREAD_SIZE*2>>> (yuv, now2DPtr);

	for (step; step < endStep; step++) {
		if (step % 2 == 0) {
			now2DPtr = gpu2DPtr0;
			next2DPtr = gpu2DPtr1;
		}
		else {
			now2DPtr = gpu2DPtr1;
			next2DPtr = gpu2DPtr0;
		}
		printf("step:	%d / %d\n", step, TOTAL_STEP);
		fflush(stdout);
		GameOfLife << <BLOCK_SIZE, THREAD_SIZE, SHARED_SIZE >> > (now2DPtr, next2DPtr);
	}
	++(impl->t);
}


void initGrids(SyncedMemory<uint8_t>* grids0,  SyncedMemory<uint8_t>* frames) {
	cudaMemset(frames->get_gpu_wo() + W*H, 128, W*H / 2);

	uint8_t* cpuPtr0 = grids0->get_cpu_wo();
	uint8_t** array2D = (uint8_t**)malloc(GRID_H * sizeof(uint8_t*));
	
	for (int i = 0; i < GRID_H; i++) {
		array2D[i] = cpuPtr0 + (i*GRID_W);
	}

	memset(cpuPtr0, DEAD, GRID_SIZE * sizeof(uint8_t));
	
	readRLE("pattern/3EngineCordRake.txt", array2D, 720, 130);

}

// read RLE file
// RLE detailhttp://conwaylife.com/wiki/RLE
void readRLE(char* fileName, uint8_t** array2D, int startX, int startY) {
	int i = startX;
	int j = startY;
	char buff[1024];
	char c;
	bool write = false;
	int value = 0, state;
	int fileX, fileY, count;
	FILE * pFile;
	pFile = fopen(fileName, "r");
	if (pFile == NULL) perror("Error opening file");
	else {
		// print comment
		do {
			fgets(buff, 1024, pFile);
		} while (buff[0]=='#');
		count = 0;
		// read girds info
		do {
			if (buff[count] == 'x') {
				count += 4;
				fileX = atoi(&buff[count]);
				value = 0;
			}
			else if (buff[count] == 'y') {
				count += 4;
				fileY = atoi(&buff[count]);
				value = 0;
				break;
			}
			count++;
		} while (count<1024);
		if (fileX + startX > GRID_W || fileY + startY > GRID_H) {
			printf("pattern size out of boundary, FAIL TO READ FILE\n");
			return;
		}
		// read cell data
		do {
			c = fgetc(pFile);

			switch (c) {
			case 'o':
				state = LIVE;
				write = true;
				break;

			case 'b':
				state = DEAD;
				write = true;
				break;

			case '$':
				if (value != 0) {
					j += value;
					value = 0;
				}
				else {
					j++;
				}
				i = startX;
				break;

			case '\n':
			case EOF:
				break;

			default:
				value *= 10;
				value += atoi(&c);
				break;
			}
			if (write) {
				if (value == 0) value = 1;
				for (count = 0; count < value; count++) {
					array2D[j][i] = state;
					i++;
				}
				write = false;
				value = 0;
			}
		} while (c != '!' && c != EOF);
		fclose(pFile);
	}
}

__global__ void initGPU(uint8_t* gridsPtr, uint8_t** grids2DPtr) {
	for (int i = 0; i < GRID_H; i++) {
		grids2DPtr[i] = &gridsPtr[GRID_W*i];
	}
}


/*
Do the Game of life iteration based on GPU.
*/
__global__ void GameOfLife(uint8_t** now2DPtr, uint8_t** next2DPtr) {
	extern __shared__ uint8_t stateArray[];
	
	int iStart = threadIdx.x * GRID_W/ blockDim.x;
	int iEnd = (threadIdx.x+1) * GRID_W / blockDim.x;
	int jStart = blockIdx.x * GRID_H / gridDim.x;
	int jEnd = (blockIdx.x+1) * GRID_H / gridDim.x;
	int sharedH = jEnd - jStart+2;

	int i, j, x, y;
	
	// Create 2D array
	uint8_t** shared2DPtr = (uint8_t**) malloc(sharedH*sizeof(uint8_t*));
	for (i = 0; i < sharedH; i++) {
		shared2DPtr[i] = &stateArray[i*GRID_W];
	}
	y = 0;
	x = iStart;
	// read top row
	if (jStart != 0) {
		for (i = iStart; i < iEnd; i++) {
			shared2DPtr[y][x] = now2DPtr[jStart - 1][i];
			x++;
		}
	}
	else {
		for (i = iStart; i < iEnd; i++) {
			shared2DPtr[y][x] = 0;
			x++;
		}
	}
	y++;
	x = iStart;
	
	// read center area
	for (j = jStart; j < jEnd; j++) {
		for (i = iStart; i < iEnd; i++) {
			shared2DPtr[y][x] = now2DPtr[j][i];
			x++;
		}
		x = iStart;
		y++;
	}

	// read bottom row
	if (jEnd != GRID_H) {
		for (i = iStart; i < iEnd; i++) {
			shared2DPtr[y][x] = now2DPtr[jEnd][i];
			x++;
		}
	} 
	else {
		for (i = iStart; i < iEnd; i++) {
			shared2DPtr[y][x] = 0;
			x++;
		}
	}
	__syncthreads();
	
	int value = 0, state = DEAD;
	y = 1;
	x = iStart;
	for (i = iStart; i < iEnd; i++) {
		for (j = jStart; j < jEnd; j++) {
			state = shared2DPtr[y][x];
			
			value += shared2DPtr[y - 1][x];
			value += shared2DPtr[y + 1][x];
			if (x != 0) {
				value += shared2DPtr[y][x-1];
				value += shared2DPtr[y - 1][x - 1];
				value += shared2DPtr[y + 1][x - 1];
			}
			if (x != (GRID_W - 1)) {
				value += shared2DPtr[y][x + 1];
				value += shared2DPtr[y - 1][x + 1];
				value += shared2DPtr[y + 1][x + 1];
			}

			state = ((value == 2 || value == 3) && state) || ((value == 3) && (!state)) ? LIVE:DEAD;
			
			next2DPtr[j][i] = state;
			value = 0;
			y++;
		}
		y = 1;
		x++;
	}
	free(shared2DPtr);

}


//Making video
__global__ void Draw(uint8_t* yuv, uint8_t** grids2DPtr)
{
	int blkId = blockIdx.x;
	int iStart = threadIdx.x * GRID_W/ blockDim.x;
	int iEnd = (threadIdx.x+1) * GRID_W / blockDim.x;
	unsigned int i, j, k, l;
	unsigned int	count;
	int value;
	int rowStart = blkId * GRID_H / BLOCK_SIZE;
	int rowEnd = (blkId+1) * GRID_H / BLOCK_SIZE;
	
	// Draw black value
	
	for (j = rowStart; j < rowEnd; j++) {
		count = j*PIXEL_H*W + iStart*PIXEL_W;
		//for (i = 0; i < GRID_W; i++) {
		for (i = iStart; i < iEnd; i++) {
			value = (grids2DPtr[j][i] == LIVE) ? 255 : 0;
			
			for (l = 0; l < PIXEL_H; l++) {
				for (k = 0; k < PIXEL_W; k++) {
					yuv[count + l*W + k] = value;
				}
			}
			count += PIXEL_W;
		}
		//count += (PIXEL_H-1)* W;
	}
}
