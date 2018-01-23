
#ifndef __LIB_CU_H__
#define __LIB_CU_H__

typedef double real;

__declspec(dllexport) void calculate_layer_output(real input[], real weight[], real bias[], real output[], size_t input_num, size_t output_num, real merge[]);
__declspec(dllexport) void calculate_layer_grad(real output[], real target[], real grad[], real study_rate, size_t output_num);
__declspec(dllexport) void calculate_layer_grad(real gradin[], real weight[], real grad[], size_t input_num, size_t output_num, real merge[]);
__declspec(dllexport) void calculate_layer_train(real input[], real grad[], real weight[], real bias[], size_t input_num, size_t output_num, real study_rate);

enum enum_translate_type
{
	float_to_double,
	double_to_float,
	real_to_double,
	real_to_float,
	double_to_real,
	float_to_real, 
};
__declspec(dllexport) void translate_data_format(void *dst, const void *src, size_t pixelNum, enum_translate_type type);

__declspec(dllexport) void set_value(real data[], size_t count, real val);
__declspec(dllexport) void set_rand_value(real data[], size_t count);

#include <Windows.h>
#include <gl\GL.h>
#include <gl\GLU.h>
__declspec(dllexport) bool gl_set_texture(GLuint texIdx, const void *data, size_t size, char *errmsg = nullptr);

__declspec(dllexport) void set_cuda_device(int dev);

__declspec(dllexport) void *cuda_malloc(size_t size);
__declspec(dllexport) void *cuda_malloc_host(size_t size);
__declspec(dllexport) void cuda_free(void *ptr);

enum enum_cuda_memcpy_direction
{
	host_to_host, host_to_device, device_to_host, device_to_device, 
};
__declspec(dllexport) void cuda_memcpy(void *dst, const void *src, size_t size, enum_cuda_memcpy_direction direction);

__declspec(dllexport) const char *cuda_get_last_error();

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
		cuda_memcpy(cb, hb, sz * sizeof(T), host_to_device);
		return *this;
	}

	inline cu_array<T>& get()
	{
		cuda_memcpy(hb, cb, sz * sizeof(T), device_to_host);
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
