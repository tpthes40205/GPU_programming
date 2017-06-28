#pragma once

#include "BEMD_utils.h"
#include "Delaunay.h"





__global__ void find_max(float** output, int* count,
	int* extrema_x, int* extrema_y, float* extrema_value,
	const int width, const int height);

float BEMD(float** output, float** max_map, float** min_map, int* extrema_array,
	const int width, const int height);


