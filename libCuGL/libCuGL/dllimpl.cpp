
#include <stdio.h>
#include <windows.h>

#include <gl\GL.h>
#include <gl\GLU.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")

BOOL GetGLCudaBuffer(GLuint texIdx, void **texData, cudaGraphicsResource_t *resource, char *errout) {
	cudaError_t err;

	if ((err = cudaGraphicsGLRegisterImage(resource, texIdx, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard)) != cudaSuccess) {
		if (errout) {
			sprintf(errout, "<err>%s", cudaGetErrorString(err));
		}
		return FALSE;
	}

	if ((err = cudaGraphicsMapResources(1, resource)) != cudaSuccess) {
		if (errout) {
			sprintf(errout, "<err>%s", cudaGetErrorString(err));
		}
		return FALSE;
	}

	void *textureData;
	if ((err = cudaGraphicsResourceGetMappedPointer(&textureData, 0, *resource)) != cudaSuccess) {
		if (errout) {
			sprintf(errout, "<err>%s", cudaGetErrorString(err));
		}
		return FALSE;
	}
	*texData = textureData;
	return TRUE;
}

extern "C" __declspec(dllexport)
BOOL SetTextureData(GLuint texIdx, void *data, GLuint size, char *errout) {
	cudaGraphicsResource_t resource;

	void *glData = nullptr;
	if (GetGLCudaBuffer(texIdx, &glData, &resource, errout)) {
		cudaMemcpy(glData, data, size, cudaMemcpyDeviceToDevice);
		
		cudaGraphicsUnmapResources(1, &resource);
		cudaGraphicsUnregisterResource(resource);
		return TRUE;
	}
	return FALSE;
}
