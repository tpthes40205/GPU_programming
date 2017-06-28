#include "Delaunay.h"





						////////////////////////////////////////////////
						//          DRAW TRIANGLE RELATED             //
						////////////////////////////////////////////////
__device__ int lower_div(int a, int b) {
	if (a < 0) { return (a + 1) / b - 1; }
	else { return a / b; }
}
__device__ int upper_div(int a, int b) {
	if (a > 0) { return (a - 1) / b + 1; }
	else { return a / b; }
}
__global__ void draw_triangles(float** map, int* extrema_x, int* extrema_y, float* extrema_value, int* triangles, const int* count_list) {
	int points[3];
	float  a[3];		// a0*x + a1*y + a2 = z
	int max_height;
	bool clockwise;
	int *left_bound, *right_bound;
	int index;
	for (index = 0; index < count_list[COUNT_TRI]; index++) {
		sort_points(triangles, index, extrema_y, points);
		find_direction(extrema_x, extrema_y, points, &clockwise);
		max_height = extrema_y[points[0]] - extrema_y[points[2]];
		left_bound = (int*)malloc((max_height + 1) * sizeof(int));
		right_bound = (int*)malloc((max_height + 1) * sizeof(int));

		int i, j, h, d, x0;
		// right bound
		if (clockwise) {
			h = max_height;
			d = extrema_x[points[0]] - extrema_x[points[2]];
			x0 = extrema_x[points[2]];
			if (h == 0) { printf("tiangles: %d) h = 0\n", index); asm("trap;"); }
			for (i = 0; i <= h; i++) {
				right_bound[i] = lower_div(d*i, h) + x0;
			}
		}
		else {
			h = extrema_y[points[1]] - extrema_y[points[2]];
			d = extrema_x[points[1]] - extrema_x[points[2]];
			x0 = extrema_x[points[2]];
			i = 0;
			if (h == 0) {
				right_bound[i] = extrema_x[points[1]];
				i++;
			}
			else {
				for (i; i <= h; i++) {
					right_bound[i] = lower_div(d*i, h) + x0;
				}
			}

			h = extrema_y[points[0]] - extrema_y[points[1]];
			d = extrema_x[points[0]] - extrema_x[points[1]];
			x0 = extrema_x[points[1]];
			for (j = 1; j <= h; j++) {
				right_bound[i] = lower_div(d*j, h) + x0;
				i++;
			}
		}

		// left bound
		if (!clockwise) {
			h = max_height;
			d = extrema_x[points[0]] - extrema_x[points[2]];
			x0 = extrema_x[points[2]];
			if (h == 0) { printf("tiangles: %d) h = 0\n", index); asm("trap;"); }
			for (i = 0; i <= h; i++) {
				left_bound[i] = upper_div(d*i, h) + x0;
			}
		}
		else {
			h = extrema_y[points[1]] - extrema_y[points[2]];
			d = extrema_x[points[1]] - extrema_x[points[2]];
			x0 = extrema_x[points[2]];
			i = 0;
			if (h == 0) {
				left_bound[i] = extrema_x[points[1]];
				i++;
			}
			else {
				for (i; i <= h; i++) {
					left_bound[i] = upper_div(d*i, h) + x0;
				}
			}

			h = extrema_y[points[0]] - extrema_y[points[1]];
			d = extrema_x[points[0]] - extrema_x[points[1]];
			x0 = extrema_x[points[1]];
			for (j = 1; j <= h; j++) {
				left_bound[i] = upper_div(d*j, h) + x0;
				i++;
			}
		}
		cramer(extrema_x, extrema_y, extrema_value, points, a);
		j = extrema_y[points[2]];
		for (h = 0; h <= max_height; h++) {
			for (i = left_bound[h]; i <= right_bound[h]; i++) { map[j][i] = a[0] * i + a[1] * j + a[2]; }
			j++;
		}
		free(left_bound);
		free(right_bound);
	}
}
__device__ void sort_points(const int* triangles, const int index, const int* extrema_y, int* points) {
	int p1 = triangles[index * 3 + 0], p2 = triangles[index * 3 + 1], p3 = triangles[index * 3 + 2];
	if (extrema_y[p1] < extrema_y[p2]) {
		points[0] = p2;
		points[1] = p1;
	}
	else {
		points[0] = p1;
		points[1] = p2;
	}
	if (extrema_y[p3] <= extrema_y[points[1]]) {
		points[2] = p3;
	}
	else {
		points[2] = points[1];
		points[1] = p3;
		if (extrema_y[points[1]] > extrema_y[points[0]]) {
			int temp = points[0];
			points[0] = points[1];
			points[1] = temp;
		}
	}
}
__device__ void find_direction(const int* extrema_x, const int* extrema_y, const int* points, bool* clockwise) {
	float vec_A[2], vec_B[2];
	float z;
	vec_A[0] = extrema_x[points[2]] - extrema_x[points[0]];
	vec_A[1] = extrema_y[points[2]] - extrema_y[points[0]];
	vec_B[0] = extrema_x[points[1]] - extrema_x[points[0]];
	vec_B[1] = extrema_y[points[1]] - extrema_y[points[0]];
	z = vec_A[0] * vec_B[1] - vec_A[1] * vec_B[0];
	if (z < 0) { *clockwise = true; }
	else { *clockwise = false; }
	
}
__device__ void cramer(const int* extrema_x, const int* extrema_y, const float* extrema_value, const int* points, float* a) {
	float delta, delta_x, delta_y, delta_z;

	delta = extrema_x[points[0]] * extrema_y[points[1]]
		+ extrema_x[points[1]] * extrema_y[points[2]]
		+ extrema_x[points[2]] * extrema_y[points[0]];
	delta -= (extrema_x[points[0]] * extrema_y[points[2]]
		+ extrema_x[points[1]] * extrema_y[points[0]]
		+ extrema_x[points[2]] * extrema_y[points[1]]);


	delta_x = extrema_value[points[0]] * extrema_y[points[1]]
		+ extrema_value[points[1]] * extrema_y[points[2]]
		+ extrema_value[points[2]] * extrema_y[points[0]];
	delta_x -= (extrema_value[points[0]] * extrema_y[points[2]]
		+ extrema_value[points[1]] * extrema_y[points[0]]
		+ extrema_value[points[2]] * extrema_y[points[1]]);

	delta_y = extrema_x[points[0]] * extrema_value[points[1]]
		+ extrema_x[points[1]] * extrema_value[points[2]]
		+ extrema_x[points[2]] * extrema_value[points[0]];
	delta_y -= (extrema_x[points[0]] * extrema_value[points[2]]
		+ extrema_x[points[1]] * extrema_value[points[0]]
		+ extrema_x[points[2]] * extrema_value[points[1]]);

	delta_z = extrema_x[points[0]] * extrema_y[points[1]] * extrema_value[points[2]]
		+ extrema_x[points[1]] * extrema_y[points[2]] * extrema_value[points[0]]
		+ extrema_x[points[2]] * extrema_y[points[0]] * extrema_value[points[1]];
	delta_z -= (extrema_x[points[0]] * extrema_y[points[2]] * extrema_value[points[1]]
		+ extrema_x[points[1]] * extrema_y[points[0]] * extrema_value[points[2]]
		+ extrema_x[points[2]] * extrema_y[points[1]] * extrema_value[points[0]]);

	a[0] = delta_x / delta;
	a[1] = delta_y / delta;
	a[2] = delta_z / delta;
}


						////////////////////////////////////////////////
						//             DELAUNAY RELATED               //
						////////////////////////////////////////////////
__device__ void write_triangle(int* triangles, const int index, const int p0, const int p1, const int p2) {
	triangles[index * 3] = p0;
	triangles[index * 3 + 1] = p1;
	triangles[index * 3 + 2] = p2;
}
__device__ void write_triangle(int* dst, const int* src) {
	dst[0] = src[0];
	dst[1] = src[1];
	dst[2] = src[2];
}

__device__ int sign(const int p, const int t0, const int t1,const int* x, const int* y) {
	return (x[t0] - x[p]) * (y[t1] - y[p]) - (x[t1] - x[p]) * (y[t0] - y[p]);
}

__device__ int distance(const int A, const int B, const int* x, const int* y) {
	int dx = x[A] - x[B], dy = y[A] - y[B];
	return dx * dx + dy * dy;
}
__device__ point_status in_triangle(const int p, const int* t, const int* x, const int* y) {
	const int A = t[0];
	const int B = t[1];
	const int C = t[2];
	int i;
	int s[3];
	s[0] = sign(p, A, B, x, y);
	s[1] = sign(p, B, C, x, y);
	s[2] = sign(p, C, A, x, y);
	for (i = 0; i < 3; i++) {
		if (s[i] == 0) {
			int L = distance(t[i], t[(i + 1) % 3], x, y);
			if (distance(p, t[i], x, y) < L && distance(p, t[(i + 1) % 3], x, y) < L) {
				return ON_EDGE;
			}
		}
	}

	// if sign(PAxPB)==sign(PBxPC)==sign(PCxPA) then INSIDE
	if ((s[0] * s[1]) > 0 && (s[1] * s[2]) > 0) { return INSIDE; }
	else { return OUTSIDE; }
}
__device__ int on_which_edge(const int p, const int* t, const int* extrema_x, const int* extrema_y) {
	const int A = t[0];
	const int B = t[1];
	const int C = t[2];
	int s[3];
	int i;
	s[0] = sign(p, A, B, extrema_x, extrema_y);
	s[1] = sign(p, B, C, extrema_x, extrema_y);
	s[2] = sign(p, C, A, extrema_x, extrema_y);
	for (i = 0; i < 3; i++) {
		if (s[i] == 0) { return i; }
	}
	
	printf("point: %d not found on edge\n", p);
	printf("%d, %d, %d\n", A, B, C);
	return -1;
}
__device__ void find_triangle(const int point, const int* hist_graph, const int index, const int* extrema_x, const int* extrema_y, int* find_info) {
	int this_address, child_count, child_index, i;
	this_address = index * HIST_COLUMN;
	child_count = hist_graph[this_address + HIST_CHILD];

	if (child_count == 0) {
		find_info[FIND_TRI] = hist_graph[this_address + HIST_HIST_0];
		find_info[FIND_STAUS] = in_triangle(point, &hist_graph[this_address], extrema_x, extrema_y);
	}
	else {
		for (i = 0; i < child_count; i++) {
			child_index = hist_graph[this_address + HIST_HIST_0 + i];
			if (in_triangle(point, &hist_graph[child_index * HIST_COLUMN], extrema_x, extrema_y) != OUTSIDE) {
				find_triangle(point, hist_graph, child_index, extrema_x, extrema_y, find_info);
				i = 999;
			}
		}
		if (i != 1000) { printf("point: %d, not found in history: %d\n", point, this_address / HIST_COLUMN); }
	}
}

__device__ void set_circum_det(const int* extrema_x, const int* extrema_y, const int A, const int D, long long int* AD) {
	int Ax = extrema_x[A], Ay = extrema_y[A], Dx = extrema_x[D], Dy = extrema_y[D];
	AD[0] =  Ax - Dx;
	AD[1] =  Ay - Dy;
	AD[2] = (Ax * Ax - Dx * Dx) + (Ay * Ay - Dy * Dy);
}
__device__ bool in_circum_circle(const int* extrema_x, const int* extrema_y, const int A, const int B, const int C, const int D) {
	long long int AD[3], CD[3], BD[3]; //transfer into counter-clockwise
	long long int det_value;
	set_circum_det(extrema_x, extrema_y, A, D, AD);
	set_circum_det(extrema_x, extrema_y, C, D, CD);
	set_circum_det(extrema_x, extrema_y, B, D, BD);
	det_value = AD[0] * CD[1] * BD[2] 
		+ AD[2] * CD[0] * BD[1] 
		+ AD[1] * CD[2] * BD[0]
		- AD[2] * CD[1] * BD[0] 
		- AD[0] * CD[2] * BD[1]
		- AD[1] * CD[0] * BD[2];
	return det_value < 0;
}

__device__ int find_neighbor(const int* neighbors, const int index, const int target) {
	for (int i = 0; i < 3; i++) {
		if (neighbors[index * 3 + i] == target) { return i; }
	}
	printf("index: %d, neighbor: %d not found\n", index, target);
	return -1;
}
__device__ void change_neighbor(int* neighbors, const int index, const int old_t, const int new_t) {
	if(index != -1){ neighbors[index * 3 + find_neighbor(neighbors, index, old_t)] = new_t; }
}

__device__ void write_hist_child(int* hist_graph, int* hist_index, const int* triangles, const int index, const int t0) {
	int address = index *  HIST_COLUMN;
	write_triangle(&hist_graph[address], &triangles[t0 * 3]);
	hist_graph[address + HIST_CHILD] = 0;
	hist_graph[address + HIST_HIST_0] = t0;
	hist_index[t0] = index;
}
__device__ void exchange_hist(int* hist_graph, int* hist_index, const int* triangles, const int t0, const int t1, int* count_list) {
	int add_0, add_1;
	add_0 = hist_index[t0] * HIST_COLUMN;
	add_1 = hist_index[t1] * HIST_COLUMN;
	hist_graph[add_0 + HIST_CHILD] = hist_graph[add_1 + HIST_CHILD] = 2;
	hist_graph[add_0 + HIST_HIST_0] = hist_graph[add_1 + HIST_HIST_0] = count_list[COUNT_HIST]++;
	hist_graph[add_0 + HIST_HIST_1] = hist_graph[add_1 + HIST_HIST_1] = count_list[COUNT_HIST]++;
	write_hist_child(hist_graph, hist_index, triangles, hist_graph[add_0 + HIST_HIST_0], t0);
	write_hist_child(hist_graph, hist_index, triangles, hist_graph[add_0 + HIST_HIST_1], t1);
}
__device__ void write_hist(int* hist_graph, int* hist_index, const int* triangles, const int t0, const int t1, int* count_list) {

	int address = hist_index[t0] * HIST_COLUMN;
	hist_graph[address + HIST_CHILD] = 2;
	hist_graph[address + HIST_HIST_0] = count_list[COUNT_HIST]++;
	hist_graph[address + HIST_HIST_1] = count_list[COUNT_HIST]++;
	
	write_hist_child(hist_graph, hist_index, triangles, hist_graph[address + HIST_HIST_0], t0);
	write_hist_child(hist_graph, hist_index, triangles, hist_graph[address + HIST_HIST_1], t1);
	
}
__device__ void write_hist(int* hist_graph, int* hist_index, const int* triangles, const int t0, const int t1, const int t2, int* count_list) {
	
	int address = hist_index[t0] *  HIST_COLUMN;
	hist_graph[address + HIST_CHILD] = 3;
	hist_graph[address + HIST_HIST_0] = count_list[COUNT_HIST]++;
	hist_graph[address + HIST_HIST_1] = count_list[COUNT_HIST]++;
	hist_graph[address + HIST_HIST_2] = count_list[COUNT_HIST]++;

	write_hist_child(hist_graph, hist_index, triangles, hist_graph[address + HIST_HIST_0], t0);
	write_hist_child(hist_graph, hist_index, triangles, hist_graph[address + HIST_HIST_1], t1);
	write_hist_child(hist_graph, hist_index, triangles, hist_graph[address + HIST_HIST_2], t2);
}

__device__ void flip(const int tri_0, const int tri_1, int* triangles, int* neighbors, int* hist_graph, int* hist_index, int* count_list,
	const int* extrema_x, const int* extrema_y) {
	if (tri_0 != -1 && tri_1 != -1) {
		int tri0_nb, tri1_nb;
		int t0[3], t1[3];
		bool in_circle;
		write_triangle(t0, &triangles[tri_0 * 3]);
		write_triangle(t1, &triangles[tri_1 * 3]);
		tri0_nb = find_neighbor(neighbors, tri_0, tri_1);
		tri1_nb = find_neighbor(neighbors, tri_1, tri_0);
		in_circle = in_circum_circle(extrema_x, extrema_y, t0[(tri0_nb + 2) % 3], t0[tri0_nb], t0[(tri0_nb + 1) % 3], t1[(tri1_nb + 2) % 3]);
		if (in_circle) {

			//printf("flip tri: %d, %d\n", tri_0, tri_1);
			//printf("tri %d) %d, %d, %d\n", tri_0, triangles[tri_0 * 3], triangles[tri_0 * 3 + 1], triangles[tri_0 * 3 + 2]);
			//printf("tri %d) %d, %d, %d\n", tri_1, triangles[tri_1 * 3], triangles[tri_1 * 3 + 1], triangles[tri_1 * 3 + 2]);

			t0[(tri0_nb + 1) % 3] = t1[(tri1_nb + 2) % 3];
			t1[(tri1_nb + 1) % 3] = t0[(tri0_nb + 2) % 3];
			neighbors[tri_0 * 3 + tri0_nb] = neighbors[tri_1 * 3 + (tri1_nb + 1) % 3];
			neighbors[tri_1 * 3 + tri1_nb] = neighbors[tri_0 * 3 + (tri0_nb + 1) % 3];
			neighbors[tri_0 * 3 + (tri0_nb + 1) % 3] = tri_1;
			neighbors[tri_1 * 3 + (tri1_nb + 1) % 3] = tri_0;
			change_neighbor(neighbors, neighbors[tri_0 * 3 + tri0_nb], tri_1, tri_0);
			change_neighbor(neighbors, neighbors[tri_1 * 3 + tri1_nb], tri_0, tri_1);
			write_triangle(&triangles[tri_0 * 3],t0);
			write_triangle(&triangles[tri_1 * 3],t1);
			exchange_hist(hist_graph, hist_index, triangles, tri_0, tri_1, count_list);

			//printf("After flip\n");
			//printf("tri %d) %d, %d, %d\n", tri_0, triangles[tri_0 * 3], triangles[tri_0 * 3 + 1], triangles[tri_0 * 3 + 2]);
			//printf("tri %d) %d, %d, %d\n", tri_1, triangles[tri_1 * 3], triangles[tri_1 * 3 + 1], triangles[tri_1 * 3 + 2]);

			flip(tri_0, neighbors[tri_0 * 3 + tri0_nb], triangles, neighbors, hist_graph, hist_index, count_list, extrema_x, extrema_y);
			flip(tri_1, neighbors[tri_1 * 3 + tri1_nb], triangles, neighbors, hist_graph, hist_index, count_list, extrema_x, extrema_y);
		}
	}
}
__device__ int insert_on_edge(const int point, const int tri_0, const int tri_1, const int tri_0_nb,const int mark, int* triangles, int* neighbors, 
	int* hist_graph, int* hist_index, int* count_list, const int* extrema_x, const int* extrema_y) {
	int new_0, new_1;
	int A, B, C, N_B;
	A = triangles[tri_0 * 3 + tri_0_nb];
	B = triangles[tri_0 * 3 + (tri_0_nb + 1) % 3];
	C = triangles[tri_0 * 3 + (tri_0_nb + 2) % 3];
	N_B = neighbors[tri_0 * 3 + (tri_0_nb + 1) % 3];
	new_0 = count_list[COUNT_TRI]++;
	if (tri_1 != -1) { new_1 = new_0 + mark; }
	else { new_1 = tri_1; }
	
	write_triangle(triangles, new_0, point, B, C);
	write_triangle(neighbors, new_0, tri_1, N_B, tri_0);
	change_neighbor(neighbors, N_B, tri_0, new_0);
	
	triangles[tri_0 * 3 + (tri_0_nb + 1) % 3] = point;
	neighbors[tri_0 * 3 + tri_0_nb] = new_1;
	neighbors[tri_0 * 3 + (tri_0_nb + 1) % 3] = new_0;
	
	write_hist(hist_graph, hist_index, triangles, tri_0, new_0, count_list);
	return new_0;
}
__device__ void insert_point(const int point, const int* find_info, int* triangles, int* neighbors, int* hist_graph, int* hist_index,  int* count_list,
	const int* extrema_x, const int* extrema_y) {
	int id;
	int tri_0 = find_info[FIND_TRI];
	int A, B, C, N_A, N_B, N_C, new_1, new_2;
	if (find_info[FIND_STAUS] == INSIDE) {
		
		A = triangles[tri_0 * 3];
		B = triangles[tri_0 * 3 + 1];
		C = triangles[tri_0 * 3 + 2];
		N_A = neighbors[tri_0 * 3];
		N_B = neighbors[tri_0 * 3 + 1];
		N_C = neighbors[tri_0 * 3 + 2];
		new_1 = count_list[COUNT_TRI]++;
		new_2 = count_list[COUNT_TRI]++;

		//printf("point: %d, insert in tri: %d\n", point, tri_0);
		//printf("tri %d) %d, %d, %d\n", tri_0, triangles[tri_0 * 3], triangles[tri_0 * 3 + 1], triangles[tri_0 * 3 + 2]);

		write_triangle(triangles, tri_0, A, B, point);
		write_triangle(triangles, new_1, B, C, point);
		write_triangle(triangles, new_2, C, A, point);
		write_triangle(neighbors, tri_0, N_A, new_1, new_2);
		write_triangle(neighbors, new_1, N_B, new_2, tri_0);
		write_triangle(neighbors, new_2, N_C, tri_0, new_1);
		//change_neighbor(neighbors, N_A, tri_0, tri_0);
		change_neighbor(neighbors, N_B, tri_0, new_1);
		change_neighbor(neighbors, N_C, tri_0, new_2);

		/*
		printf("After insert\n");
		printf("tri %d) %d, %d, %d\n", tri_0, triangles[tri_0 * 3], triangles[tri_0 * 3 + 1], triangles[tri_0 * 3 + 2]);
		printf("tri %d) %d, %d, %d\n", new_1, triangles[new_1 * 3], triangles[new_1 * 3 + 1], triangles[new_1 * 3 + 2]);
		printf("tri %d) %d, %d, %d\n\n", new_2, triangles[new_2 * 3], triangles[new_2 * 3 + 1], triangles[new_2 * 3 + 2]);
		*/
		write_hist(hist_graph, hist_index, triangles, tri_0, new_1, new_2, count_list);
		
		flip(tri_0, N_A, triangles, neighbors, hist_graph, hist_index, count_list, extrema_x, extrema_y);
		flip(new_1, N_B, triangles, neighbors, hist_graph, hist_index, count_list, extrema_x, extrema_y);
		flip(new_2, N_C, triangles, neighbors, hist_graph, hist_index, count_list, extrema_x, extrema_y);
		
	}
	else {
		int tri_1, new_0, new_1;
		int tri_0_nb, tri_1_nb;
		tri_0_nb = on_which_edge(point, &triangles[tri_0 * 3], extrema_x, extrema_y);
		tri_1 = neighbors[tri_0 * 3 + tri_0_nb];
		
		new_0 = insert_on_edge(point, tri_0, tri_1, tri_0_nb, 1,  triangles, neighbors, hist_graph, hist_index, count_list, extrema_x, extrema_y);
		if(tri_1 != -1){ 
			tri_1_nb = on_which_edge(point, &triangles[tri_1 * 3], extrema_x, extrema_y);
			new_1 = insert_on_edge(point, tri_1, tri_0, tri_1_nb, -1, triangles, neighbors, hist_graph, hist_index, count_list, extrema_x, extrema_y);
			flip(tri_1, neighbors[tri_1*3+(tri_1_nb +2)%3], triangles, neighbors, hist_graph, hist_index, count_list, extrema_x, extrema_y);
			flip(new_1, neighbors[new_1 * 3 + 1], triangles, neighbors, hist_graph, hist_index, count_list, extrema_x, extrema_y);
		}
		flip(tri_0, neighbors[tri_0 * 3 + (tri_0_nb + 2) % 3], triangles, neighbors, hist_graph, hist_index, count_list, extrema_x, extrema_y);
		flip(new_0, neighbors[new_0 * 3 + 1], triangles, neighbors, hist_graph, hist_index, count_list, extrema_x, extrema_y);
	}
}

__global__ void initialize(float** map,
	const int* extrema_x, const int* extrema_y, const float* extrema_value,
	int* triangles, int* neighbors, int* history_index, int* history_graph, int* count_list,
	const int width, const int height){
	
	history_graph[0 * HIST_COLUMN + HIST_CHILD] = 2;
	history_graph[0 * HIST_COLUMN + HIST_HIST_0] = 1;
	history_graph[0 * HIST_COLUMN + HIST_HIST_1] = 2;

	write_triangle(triangles, 0, 0, 1, 2);
	write_triangle(neighbors, 0, -1, 1, -1);
	history_graph[1 * HIST_COLUMN + HIST_PT_0] = 0;
	history_graph[1 * HIST_COLUMN + HIST_PT_1] = 1;
	history_graph[1 * HIST_COLUMN + HIST_PT_2] = 2;
	history_graph[1 * HIST_COLUMN + HIST_CHILD] = 0;
	history_graph[1 * HIST_COLUMN + HIST_HIST_0] = 0;
	history_index[0] = 1;
	

	write_triangle(triangles, 1, 1, 3, 2);
	write_triangle(neighbors, 1, -1, -1, 0);
	history_graph[2 * HIST_COLUMN + HIST_PT_0] = 1;
	history_graph[2 * HIST_COLUMN + HIST_PT_1] = 3;
	history_graph[2 * HIST_COLUMN + HIST_PT_2] = 2;
	history_graph[2 * HIST_COLUMN + HIST_CHILD] = 0;
	history_graph[2 * HIST_COLUMN + HIST_HIST_0] = 1;
	history_index[1] = 2;
	
	count_list[COUNT_TRI] = 2;
	count_list[COUNT_HIST] = 3;

	int i,j;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			map[i][j] = 255;
		}
	}
}
__global__ void incremental_construction(int* points_count,const int mode, const int* extrema_x, const int* extrema_y,
	int* triangles, int* neighbors, int* hist_index, int* hist_graph, int* count_list) {
	int find_info[FIND_SIZE];
	int i, k, pt_count;

	pt_count = points_count[mode];
	
	/*
	printf("Points:\n");
	for (i = 0; i < pt_count; i++) {
		printf("%d, %d\n", extrema_x[i], extrema_y[i]);
	}
	printf("\n");
	*/
	//printf("%d, %d, %d\n", points_count[mode], count_list[COUNT_TRI], count_list[COUNT_HIST]);

	for (i = 4; i <pt_count; i++) {
		find_triangle(i, hist_graph, 0, extrema_x, extrema_y, find_info);
		//printf("point: %d, in tri: %d, status: %d\n", i, find_info[FIND_TRI], find_info[FIND_STAUS]);
		k = find_info[FIND_TRI];
		insert_point(i, find_info, triangles, neighbors, hist_graph, hist_index, count_list, extrema_x, extrema_y);
		if (count_list[COUNT_TRI] > TRIANGLE_SIZE) {
			printf("COUNT_TRI out of TRIANGLE_SIZE\n");
			asm("trap;"); 
		}
		if (count_list[COUNT_HIST] > HIST_SIZE) {
			printf("COUNT_HIST out of HIST_SIZE\n");
			asm("trap;");
		}
	}

	/*
	printf("\nTriangles:\n");
	for (i = 0; i < count_list[COUNT_TRI]; i++) {
		printf("%d) %d, %d, %d\n", i, triangles[i * 3], triangles[i * 3 + 1], triangles[i * 3 + 2]);
	}
	printf("\nNeighbors:\n");
	for (i = 0; i < count_list[COUNT_TRI]; i++) {
		printf("%d) %d, %d, %d\n", i, neighbors[i * 3], neighbors[i * 3 + 1], neighbors[i * 3 + 2]);
	}
	printf("\nHistory Graph:\n");
	for (i = 0; i < count_list[COUNT_HIST]; i++) {
		int address = i * HIST_COLUMN;
		printf("%d) %d, %d, %d |%d| %d, %d, %d\n", i, hist_graph[address + HIST_PT_0], hist_graph[address + HIST_PT_1], hist_graph[address + HIST_PT_2],
			hist_graph[address + HIST_CHILD], hist_graph[address + HIST_HIST_0], hist_graph[address + HIST_HIST_1], hist_graph[address + HIST_HIST_2]);
	} 
	*/
}


void write_csv(const char* file_path, int* extrema_x, int* extrema_y, int* triangles, int* count_list, int* points_count, int mode) {
	int *host_x, *host_y, *host_tri, *host_tri_count, *host_pt_count;
	host_x = (int*)malloc(EXTREMA_SIZE * sizeof(int));
	host_y = (int*)malloc(EXTREMA_SIZE * sizeof(int));
	host_tri = (int*)malloc(TRIANGLE_SIZE * 3 * sizeof(int));
	host_tri_count = (int*)malloc(COUNT_SIZE * sizeof(int));
	host_pt_count = (int*)malloc(2 * sizeof(int));
	cudaMemcpy(host_x, extrema_x, EXTREMA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_y, extrema_y, EXTREMA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_tri, triangles, TRIANGLE_SIZE * 3 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_tri_count, count_list, COUNT_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_pt_count, points_count, 2 * sizeof(int), cudaMemcpyDeviceToHost);

	FILE*  fp;
	int i;
	fp = fopen(file_path, "w");
	fprintf(fp, "%d\n", host_pt_count[mode]);
	for (i = 0; i < host_pt_count[mode]; i++) {
		fprintf(fp, "%d, %d\n", host_x[i], host_y[i]);
	}

	fprintf(fp, "%d\n", host_tri_count[COUNT_TRI]);
	for (i = 0; i <  host_tri_count[COUNT_TRI]; i++) {
		fprintf(fp, "%d, %d, %d\n", host_tri[i * 3], host_tri[i * 3 + 1], host_tri[i * 3 + 2]);
	}
	fclose(fp);
	free(host_x);
	free(host_y);
	free(host_tri);
	free(host_tri_count);
}


void Delaunay_triangle(float** map, int* points_count, int mode,
	int* extrema_x, int* extrema_y, float* extrema_value,
	const int width, const int height){
	int *triangles, *neighbors, *hist_index ,*hist_graph, *count_list;

	cudaMalloc(&triangles, TRIANGLE_SIZE * 3 * sizeof(int));
	cudaMalloc(&neighbors, TRIANGLE_SIZE * 3 * sizeof(int));
	cudaMalloc(&hist_index, TRIANGLE_SIZE * sizeof(int));
	cudaMalloc(&hist_graph, HIST_SIZE * HIST_COLUMN * sizeof(int));
	cudaMalloc(&count_list, COUNT_SIZE * sizeof(int));

	initialize<<<1,1>>>(map, extrema_x, extrema_y, extrema_value, triangles, neighbors, hist_index, hist_graph, count_list, width, height);

	incremental_construction<<<1,1>>>(points_count, mode, extrema_x, extrema_y, triangles, neighbors, hist_index, hist_graph, count_list);
	cudaFree(neighbors);
	cudaFree(hist_index);
	cudaFree(hist_graph);

	draw_triangles <<<1,1>>> (map, extrema_x, extrema_y, extrema_value, triangles, count_list);
	check("error in drawing triangles");

	//write_csv("figures/result.csv", extrema_x, extrema_y, triangles,count_list, points_count, mode);

	cudaFree(count_list);
	cudaFree(triangles);
}