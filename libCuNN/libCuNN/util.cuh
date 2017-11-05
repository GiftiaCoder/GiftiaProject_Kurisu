
#ifndef _UTIL_CUH_
#define _UTIL_CUH_

#include "init.cuh"

#include <iostream>
#define PRINT_CUDA_ERROR() { std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl; }

template<count c>
struct bit_count {
	enum { val = bit_count<(c >> 1)>::val + 1 };
};

template<>
struct bit_count<1> {
	enum { val = 0 };
};

template<count c>
struct bit_mask {
	enum { val = (1 << bit_count<c>::val) - 1 };
};

template<typename T>
class CMemoryModel {
protected:
	count len;
	T *ptr;
public:
	inline T *GetData() { return ptr; }
	inline count GetSize() { return sizeof(T) * len; }
	inline count GetNum() { return len; }
};

template<typename T> class CCudaMemory;
template<typename T> class CHostMemory;

template<typename T>
class CCudaMemory : public CMemoryModel<T> {
public:
	void Init(count size) {
		len = size;
		cudaMalloc(&ptr, sizeof(T) * size);
	}
public:
	CCudaMemory() {
		len = 0;
	}
	CCudaMemory(count size) {
		Init(size);
	}
	~CCudaMemory() {
		cudaFree(ptr);
	}

public:
	inline void SetValue(real val) {
		initSetValue(GetData(), GetNum(), val);
	}
	inline void SetRand(real max, real min) {
		initSetRand(GetData(), GetNum(), max, min, ((count) this) & 0x0FFFFFFFF);
	}
public:
	inline void CopyFrom(CCudaMemory<T> &src) {
		cudaMemcpy(GetData(), src.GetData(), GetSize(), cudaMemcpyDeviceToDevice);
	}
	inline void CopyFrom(CHostMemory<T> &src) {
		cudaMemcpy(GetData(), src.GetData(), GetSize(), cudaMemcpyHostToDevice);
	}
};

template<typename T>
class CHostMemory : public CMemoryModel<T> {
public:
	void Init(count size) {
		len = size;
		cudaMallocHost(&ptr, sizeof(T) * size);
	}
public:
	CHostMemory() {}
	CHostMemory(count size) {
		Init(size);
	}
	~CHostMemory() {
		cudaFreeHost(ptr);
	}

public:
	inline T& operator[] (count idx) {
		return ptr[idx];
	}

public:
	inline void CopyFrom(CHostMemory<T> &src) {
		cudaMemcpy(GetData(), src.GetData(), GetSize(), cudaMemcpyHostToHost);
	}
	inline void CopyFrom(CCudaMemory<T> &src) {
		cudaMemcpy(GetData(), src.GetData(), GetSize(), cudaMemcpyDeviceToHost);
	}
	inline void CopyFromCuda(real src[]) {
		cudaMemcpy(GetData(), src, GetSize(), cudaMemcpyDeviceToHost);
	}
	inline void CopyFromHost(real src[]) {
		cudaMemcpy(GetData(), src, GetSize(), cudaMemcpyHostToHost);
	}
};

#endif
