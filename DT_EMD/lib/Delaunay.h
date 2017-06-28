#pragma once

#include "BEMD_utils.h"

#define TRIANGLE_SIZE 40 * BUFSIZ
#define HIST_SIZE 120 * BUFSIZ


#define HIST_COLUMN 7
enum histoty_variables {
	HIST_PT_0 = 0,
	HIST_PT_1 = 1,
	HIST_PT_2 = 2,
	HIST_CHILD = 3,
	HIST_HIST_0 = 4,
	HIST_HIST_1 = 5,
	HIST_HIST_2 = 6
};

#define COUNT_SIZE 8
enum count_variables {
	COUNT_TRI = 0,
	COUNT_HIST = 1
};

#define FIND_SIZE 2
enum find_variables {
	FIND_TRI = 0,
	FIND_STAUS = 1,
	
};

enum point_status {
	OUTSIDE = 0,
	INSIDE = 1,
	ON_EDGE = 2
};


						////////////////////////////////////////////////
						//          DRAW TRIANGLE RELATED             //
						////////////////////////////////////////////////
__device__ int lower_div(int a, int b);
__device__ int upper_div(int a, int b);
__global__ void draw_triangles(float** map, int* extrema_x, int* extrema_y, float* extrema_value, int* triangles, const int* count_list);
__device__ void sort_points(const int* triangles, const int index, const int* extrema_y, int* points);
__device__ void find_direction(const int* extrema_x, const int* extrema_y, const int* points, bool* clockwise);
__device__ void cramer(const int* extrema_x, const int* extrema_y, const float* extrema_value, const int* points, float* a);



						////////////////////////////////////////////////
						//             DELAUNAY RELATED               //
						////////////////////////////////////////////////

__device__ point_status in_triangle(const int p, const int* t,const int* extrema_x, const int* extrema_y);




void Delaunay_triangle(float** map, int* count, int mode, int* extrema_x, int* extrema_y, float* extrema_value, const int width, const int height);