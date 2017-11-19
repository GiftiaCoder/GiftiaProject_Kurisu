
#ifndef _LIB_CUGL_H_
#define _LIB_CUGL_H_

#include <Windows.h>
#include <gl\GL.h>
#include <gl\GLU.h>

#pragma comment(lib, "libCuGL.lib")

#ifdef __cplusplus
extern "C" {
#endif

	__declspec(dllexport) BOOL SetTextureData(GLuint texIdx, void *data, GLuint size, char *errout);

	__declspec(dllexport) void *MallocCudaMemory(size_t size, char *errout);

	__declspec(dllexport) void *MallocHostMemory(size_t size, char *errout);

	__declspec(dllexport) void FreeMemory(void *ptr);

	__declspec(dllexport) void CopyDataFromHostToCuda(void *dst, void *src, size_t size, char *errout);

	__declspec(dllexport) void CopyDataFromCudaToHost(void *dst, void *src, size_t size, char *errout);

	__declspec(dllexport) void CopyDataFromCudaToCuda(void *dst, void *src, size_t size, char *errout);

#ifdef __cplusplus
}
#endif

#endif
