
#include "libcu.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <stdio.h>

static cudaError_t gl_set_texture(GLuint texIdx, const void *data, size_t size, cudaGraphicsResource_t *resource, char *errmsg)
{
	cudaError_t err;
	
	if ((err = cudaGraphicsGLRegisterImage(resource, texIdx, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard)) != cudaSuccess)
	{
		if (errmsg)
		{
			sprintf(errmsg, "<%s>%s", "cudaGraphicsGLRegisterImage", cudaGetErrorString(err));
		}
		return err;
	}

	if ((err = cudaGraphicsMapResources(1, resource)) != cudaSuccess)
	{
		if (errmsg)
		{
			sprintf(errmsg, "<%s>%s", "cudaGraphicsMapResources", cudaGetErrorString(err));
		}
		return err;
	}

	cudaArray_t dst = 0;
	if ((err = cudaGraphicsSubResourceGetMappedArray(&dst, *resource, 0, 0)) != cudaSuccess)
	{
		if (errmsg)
		{
			sprintf(errmsg, "<%s>%s", "cudaGraphicsSubResourceGetMappedArray", cudaGetErrorString(err));
		}
		return err;
	}

	//cudaChannelFormatDesc d;
	//cudaExtent e;
	//unsigned int f;
	//cudaArrayGetInfo(&d, &e, &f, dst);
	//printf("(%d, %d, %d)(%d, %d, %d, %d), %d\n", e.width, e.height, e.depth, d.x, d.y, d.z, d.w, d.f);

	if ((err = cudaMemcpyToArray(dst, 0, 0, data, size, cudaMemcpyDeviceToDevice)) != cudaSuccess)
	{
		if (errmsg)
		{
			sprintf(errmsg, "<%s>%s", "cudaMemcpyToArray", cudaGetErrorString(err));
		}
		return err;
	}

	return err;
}

bool gl_set_texture(GLuint texIdx, const void *data, size_t size, char *errmsg)
{
	cudaThreadSynchronize();

	cudaGraphicsResource_t resource = 0;
	
	bool ret = (gl_set_texture(texIdx, data, size, &resource, errmsg) == cudaSuccess);

	cudaGraphicsUnmapResources(1, &resource);
	if (resource)
	{
		cudaGraphicsUnregisterResource(resource);
	}

	return ret;
}
