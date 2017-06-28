#include "lib/SyncedMemory.h"
#include "lib/pgm.h"
#include "lib/BEMD.h"
#include <string>

using namespace std;


int main(int argc, char **argv)
{
	auto input_path = "figures/2D_sin_0.pgm";
	string output_name = "figures/test_";
	bool success;
	int width, height, channel;

	auto img = ReadNetpbm(width, height, channel, success, input_path);

	if (!success) {
		puts("Something wrong with reading the input image files.");
		abort();
	}
	float** output_2D;
	const int SIZE = width*height;
	MemoryBuffer<float> output(SIZE);
	auto output_s = output.CreateSync(SIZE);
	float *output_cpu = output_s.get_cpu_wo();

	float * data = (float*) malloc(SIZE*sizeof(float));
	copy(img.get(), img.get() + SIZE, data);
	minus_value(data, 128.0, SIZE);
	
	
	cudaMalloc(&output_2D, sizeof(float*) * height);
	set_2D<<<1,1>>> (output_2D, output_s.get_gpu_rw(), width, height);

	float **max_map_2D, **min_map_2D;
	float *max_map, *min_map;
	cudaMalloc(&max_map, sizeof(float) * SIZE);
	cudaMalloc(&min_map, sizeof(float) * SIZE);
	cudaMalloc(&max_map_2D, sizeof(float *) * height);
	cudaMalloc(&min_map_2D, sizeof(float *) * height);
	set_2D<<<1,1>>> (max_map_2D, max_map, width, height);
	set_2D<<<1,1>>>(min_map_2D, min_map, width, height);

	int* extrema_array;
	cudaMalloc(&extrema_array, 6 * EXTREMA_SIZE * sizeof(int));
	int* other_;
	
	char c[BUFSIZ];
	for (int i = 0; i < 2; i++) {
		copy(data, data + SIZE, output_s.get_cpu_wo());
		output_s.get_gpu_ro();
		BEMD(output_2D, max_map_2D, min_map_2D, extrema_array, width, height);
		
		unique_ptr<uint8_t[]> o(new uint8_t[SIZE]);
		const float *o_cpu = output_s.get_cpu_sr();
		transform(o_cpu, o_cpu + SIZE, o.get(), [](float f) -> uint8_t { return max(min(int(f + 0.5f + 128), 255), 0); });
		itoa(i, c, 10);
		string output_path = output_name + c + ".pgm";
		WritePGM(o.get(), width, height, output_path.data());

		minus_data(data, o_cpu, SIZE);
	}
	
	


	system("pause");
	
	free(data);

	cudaFree(output_2D);
	cudaFree(max_map_2D);
	cudaFree(min_map_2D);
	cudaFree(max_map);
	cudaFree(min_map);
	
	return 0;

}
