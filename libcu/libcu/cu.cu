
#include "libcu.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

void *cuda_malloc(size_t size)
{
	void *ptr = 0;
	cudaMalloc(&ptr, size);
	return ptr;
}

void *cuda_malloc_host(size_t size)
{
	void *ptr = 0;
	cudaMallocHost(&ptr, size);
	return ptr;
}

void cuda_free(void *ptr)
{
	cudaFree(ptr);
}

void cuda_host_to_host(void *dst, void *src, size_t size)
{
	cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
}

void cuda_host_to_device(void *dst, void *src, size_t size)
{
	cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void cuda_device_to_host(void *dst, void *src, size_t size)
{
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void cuda_device_to_device(void *dst, void *src, size_t size)
{
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}
