
#ifndef __LIB_CU_H__
#define __LIB_CU_H__

typedef double real;
__declspec(dllexport) void calculate_layer_output(real input[], real weight[], real bias[], real output[], size_t input_num, size_t output_num, real merge[]);
__declspec(dllexport) void calculate_layer_grad(real output[], real target[], real grad[], size_t output_num);
__declspec(dllexport) void calculate_layer_grad(real gradin[], real weight[], real grad[], size_t input_num, size_t output_num, real merge[]);
__declspec(dllexport) void calculate_layer_train(real input[], real grad[], real weight[], real bias[], size_t input_num, size_t output_num, real study_rate);

#include <Windows.h>
#include <gl\GL.h>
#include <gl\GLU.h>
__declspec(dllexport) bool gl_set_texture(GLuint texIdx, void *data, GLuint size, char *errmsg = nullptr);

__declspec(dllexport) void *cuda_malloc(size_t size);
__declspec(dllexport) void *cuda_malloc_host(size_t size);
__declspec(dllexport) void cuda_free(void *ptr);
__declspec(dllexport) void cuda_host_to_host(void *dst, void *src, size_t size);
__declspec(dllexport) void cuda_host_to_device(void *dst, void *src, size_t size);
__declspec(dllexport) void cuda_device_to_host(void *dst, void *src, size_t size);
__declspec(dllexport) void cuda_device_to_device(void *dst, void *src, size_t size);

template<typename T>
class cu_array
{
	T *cb, hb;
	size_t sz;

public:
	cu_array(size_t size) :
		sz(size),
		cb((T *)cuda_malloc(size * sizeof(T))),
		hb((T *)cuda_malloc_host(size * sizeof(T))) {}

	~cu_array()
	{
		cuda_free(cb);
		cuda_free(hb);
	}

	inline T *cbuf()
	{
		return cb;
	}

	inline T *hbuf()
	{
		return hb;
	}

	inline size_t size()
	{
		return sz;
	}

	inline cu_array<T>& set()
	{
		cuda_host_to_device(cb, hb, sz * sizeof(T));
		return *this;
	}

	inline cu_array<T>& get()
	{
		cuda_device_to_host(hb, cb, sz * sizeof(T));
		return *this;
	}

	inline T& operator[] (size_t idx)
	{
		return hb[idx];
	}

	inline T *operator() ()
	{
		return cb;
	}
};

#endif
