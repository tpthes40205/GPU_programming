#pragma once
#include <cstdint>
#include <memory>
#include "SyncedMemory.h"
#include <iostream>
using std::unique_ptr;

//GPU SETTING
#define BLOCK_SIZE  200
#define THREAD_SIZE 200
#define SHARED_SIZE 30720
//VIDEO REALTED
#define W 1280
#define H 720
#define NFRAME 200
#define TOTAL_STEP 1400

//GAME OF LIFE RELATED 
#define PIXEL_W 1
static const unsigned int PIXEL_H = PIXEL_W;
static const unsigned int PIXEL_SIZE = PIXEL_W * PIXEL_H;
static const unsigned int  GRID_W = W / PIXEL_W;
static const unsigned int  GRID_H = H / PIXEL_H;
static const unsigned int GRID_SIZE = GRID_W*GRID_H;

//RETRIEVING DATA CLASS
struct Lab1VideoInfo {
	unsigned w, h, n_frame;
	unsigned fps_n, fps_d;
};

//VIDEO DRAWING CLASS
class Lab1VideoGenerator {
	struct Impl;
	unique_ptr<Impl> impl;
public:
	Lab1VideoGenerator();
	~Lab1VideoGenerator();
	void get_info(Lab1VideoInfo &info);
	void Generate(SyncedMemory<uint8_t>* grids0, SyncedMemory<uint8_t>* grids1, SyncedMemory<uint8_t>* frames, uint8_t** gpu2DPtr0, uint8_t** gpu2DPtr1);
};

enum CellState {
	DEAD = 0,
	LIVE = 1
};

// CPU FUNCTION SET
void initGrids(SyncedMemory<uint8_t>* grids0, SyncedMemory<uint8_t>* frames);
void readRLE(char* fileName, uint8_t** array2D, int x, int y);

// GPU FUNCTION SET
__global__ void initGPU(uint8_t* gridsPtr, uint8_t** grids2DPtr);
__global__ void GameOfLife(uint8_t** now2DPtr, uint8_t** next2DPtr);
__global__ void Draw(uint8_t* yuv, uint8_t** grids2DPtr);

