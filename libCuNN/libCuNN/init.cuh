
#ifndef _INIT_CUH_
#define _INIT_CUH_

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#ifndef _CUNN_BASE_TYPE_H_
#define _CUNN_BASE_TYPE_H_
typedef float real;
typedef unsigned int count;
#endif

count cudaGetGD(count pn);

count cudaGetBD(count pn);

void initSetValue(real mem[], count num, real val);

void initSetRand(real mem[], count num, real max, real min, count seed);

#endif
