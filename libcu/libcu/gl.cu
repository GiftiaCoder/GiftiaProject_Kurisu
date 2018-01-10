
#include "libcu.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <stdio.h>

bool gl_set_texture(GLuint texIdx, void *data, GLuint size, char *errmsg)
{
	cudaError_t err;

	cudaGraphicsResource_t resource;
	if ((err = cudaGraphicsGLRegisterImage(&resource, texIdx, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard)) != cudaSuccess)
	{
		if (errmsg)
		{
			sprintf(errmsg, "<%s>%s", "cudaGraphicsGLRegisterImage", cudaGetErrorString(err));
		}
		return false;
	}

	if ((err = cudaGraphicsMapResources(1, &resource)) != cudaSuccess)
	{
		if (errmsg)
		{
			sprintf(errmsg, "<%s>%s", "cudaGraphicsMapResources", cudaGetErrorString(err));
		}
		cudaGraphicsUnregisterResource(resource);
		return false;
	}

	cudaArray_t dst;
	if ((err = cudaGraphicsSubResourceGetMappedArray(&dst, resource, 0, 0)) != cudaSuccess)
	{
		if (errmsg)
		{
			sprintf(errmsg, "<%s>%s", "cudaGraphicsSubResourceGetMappedArray", cudaGetErrorString(err));
		}
		cudaGraphicsUnmapResources(1, &resource);
		cudaGraphicsUnregisterResource(resource);
		return false;
	}

	if ((err = cudaMemcpyToArray(dst, 0, 0, data, size, cudaMemcpyDeviceToDevice)) != cudaSuccess)
	{
		if (errmsg)
		{
			sprintf(errmsg, "<%s>%s", "cudaMemcpyToArray", cudaGetErrorString(err));
		}
		cudaGraphicsUnmapResources(1, &resource);
		cudaGraphicsUnregisterResource(resource);
		return false;
	}

	cudaGraphicsUnmapResources(1, &resource);
	cudaGraphicsUnregisterResource(resource);
	return true;
}
