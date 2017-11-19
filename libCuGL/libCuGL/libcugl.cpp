
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

#ifdef __cplusplus
extern "C" {
#endif

	__declspec(dllexport) BOOL SetTextureData(GLuint texIdx, void *data, GLuint size, char *errout) {
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

	__declspec(dllexport) void *MallocCudaMemory(size_t size, char *errout) {
		void *ptr = NULL;
		cudaError_t err = cudaMalloc(&ptr, size);
		if (err != cudaSuccess) {
			if (errout) {
				sprintf(errout, "<err>%s", cudaGetErrorString(err));
			}
		}
		return ptr;
	}

	__declspec(dllexport) void *MallocHostMemory(size_t size, char *errout) {
		void *ptr = NULL;
		cudaError_t err = cudaMallocHost(&ptr, size);
		if (err != cudaSuccess) {
			if (errout) {
				sprintf(errout, "<err>%s", cudaGetErrorString(err));
			}
		}
		return ptr;
	}

	__declspec(dllexport) void FreeMemory(void *ptr) {
		cudaFree(ptr);
	}

	__declspec(dllexport) void CopyDataFromHostToCuda(void *dst, void *src, size_t size, char *errout) {
		cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			if (errout) {
				sprintf(errout, "<err>%s", cudaGetErrorString(err));
			}
		}
	}

	__declspec(dllexport) void CopyDataFromCudaToHost(void *dst, void *src, size_t size, char *errout) {
		cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			if (errout) {
				sprintf(errout, "<err>%s", cudaGetErrorString(err));
			}
		}
	}

	__declspec(dllexport) void CopyDataFromCudaToCuda(void *dst, void *src, size_t size, char *errout) {
		cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
		if (err != cudaSuccess) {
			if (errout) {
				sprintf(errout, "<err>%s", cudaGetErrorString(err));
			}
		}
	}

#ifdef __cplusplus
}
#endif
