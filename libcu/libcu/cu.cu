
#include "libcu.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

void set_cuda_device(int dev)
{
	cudaSetDevice(dev);
}

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

void cuda_memcpy(void *dst, const void *src, size_t size, enum_cuda_memcpy_direction direction)
{
	cudaMemcpy(dst, src, size, (cudaMemcpyKind)direction);
}

/*void cuda_host_to_host(void *dst, const void *src, size_t size)
{
	cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
}

void cuda_host_to_device(void *dst, const void *src, size_t size)
{
	cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void cuda_device_to_host(void *dst, const void *src, size_t size)
{
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void cuda_device_to_device(void *dst, const void *src, size_t size)
{
	cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}*/

const char *cuda_get_last_error()
{
	return cudaGetErrorString(cudaGetLastError());
}
