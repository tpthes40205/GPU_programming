#pragma once

void CountPosition1(const char *text, int *pos, int text_size);
void CountPosition2(const char *text, int *pos, int text_size);
__global__ void count_kernel(const char *text, int *pos, int text_size);
__global__ void check_kernel(int *pos, int text_size);
