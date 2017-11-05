
#include "init.cuh"

count cudaGetGD(count pn) {
	const static count max_grid_num = 32 * 1024;
	pn = ((pn - 1) >> 10) + 1;
	return pn < max_grid_num ? pn : max_grid_num;
}

count cudaGetBD(count pn) {
	return pn > 1024 ? 1024 : pn;
}

__global__ void cudaSetValue(real mem[], count num, real val) {
	count thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	count thread_num = gridDim.x * blockDim.x;

	while (thread_idx < num) {
		mem[thread_idx] = val;
		thread_idx += thread_num;
	}
}

void initSetValue(real mem[], count num, real val) {
	cudaSetValue << <cudaGetGD(num), cudaGetBD(num) >> >(mem, num, val);
}

__device__ count cudaGetRand(count seed, count max, count min) {
	const static count base = (67409 << 5) ^ (67399 << 11) ^ (67391 << 17) ^ (67369 << 23) ^ (67349 << 29);
	count val = base ^ (seed << 7) ^ (seed << 13) ^ (seed << 19) ^ (seed << 23);
	return (val % (max - min + 1)) + min;
}

__global__ void cudaSetRand(real mem[], count num, real max, real min, count seed) {
	count thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	count thread_num = gridDim.x * blockDim.x;

	while (thread_idx < num) {
		mem[thread_idx] = ((((real)cudaGetRand(seed ^ thread_idx, 100000, 0)) * (max - min)) / 100000) + min;
		thread_idx += thread_num;
	}
}

void initSetRand(real mem[], count num, real max, real min, count seed) {
	cudaSetRand << <cudaGetGD(num), cudaGetBD(num) >> >(mem, num, max, min, seed);
}
