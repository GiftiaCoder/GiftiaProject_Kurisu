
#include <device_functions.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#include "libcu.h"

__global__ void cu_multiply(real input[], real weight[], real merge[], size_t input_num, size_t weight_num)
{
	size_t thread_num = gridDim.x * blockDim.x;
	size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
	while (thread_idx < weight_num)
	{
		merge[thread_idx] = input[thread_idx % input_num] * weight[thread_idx];

		thread_idx += thread_num;
	}
}

__global__ void cu_merge_output(real merge[], size_t input_num, size_t merge_num, size_t remainder, size_t output_num)
{
	size_t oprand_num = merge_num * output_num;
	size_t thread_num = gridDim.x * blockDim.x;
	size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
	while (thread_idx < oprand_num)
	{
		size_t neuro_idx = thread_idx / merge_num;
		size_t weight_idx = thread_idx % merge_num;
		size_t base_idx = neuro_idx * input_num + weight_idx;

		merge[base_idx] += merge[base_idx + merge_num + remainder];

		thread_idx += thread_num;
	}
}

__global__ void cu_output(real merge[], real bias[], real output[], size_t input_num, size_t output_num)
{
	//const static real TWO_DIV_THREE = (real)(2.0 / 3.0);

	size_t thread_num = gridDim.x * blockDim.x;
	size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
	while (thread_idx < output_num)
	{
		real out = merge[thread_idx * input_num] + bias[thread_idx];
		output[thread_idx] = ((real)2.0 * tanh(out * 0.01));
		//output[thread_idx] = ((real)1.7159 * tanh(out * TWO_DIV_THREE));
		//real tout = ((real)1.7159 * tanh(out * TWO_DIV_THREE));
		//if (__isnan(tout))
		//{
		//	tout = out > 0 ? (real)1.7159 : (real)-1.7159;
		//}
		//output[thread_idx] = tout;

		thread_idx += thread_num;
	}
}

#define THREAD_NUM_PER_BLOCK (1024)
size_t get_block_num(size_t oprand_num)
{
	const static size_t max_block_num = 42 * 1024;
	oprand_num = oprand_num / THREAD_NUM_PER_BLOCK;
	++oprand_num;
	return oprand_num < max_block_num ? oprand_num : max_block_num;
}

void calculate_merge(real merge[], size_t block_size, size_t block_num)
{
	size_t remainder = block_size & 1;
	size_t merge_num = block_size >> 1;

	while (merge_num)
	{
		cu_merge_output<<<get_block_num(merge_num * block_num), THREAD_NUM_PER_BLOCK >>>(merge, block_size, merge_num, remainder, block_num);

		remainder = (merge_num += remainder) & 1;
		
		merge_num >>= 1; // merge_num /= 2;
	}
}

void calculate_layer_output(real input[], real weight[], real bias[], real output[], size_t input_num, size_t output_num, real merge[])
{
	cu_multiply<<<get_block_num(input_num * output_num), THREAD_NUM_PER_BLOCK >>>
		(input, weight, merge, input_num, input_num * output_num);
	calculate_merge(merge, input_num, output_num);
	cu_output<<<get_block_num(output_num), THREAD_NUM_PER_BLOCK >>>
		(merge, bias, output, input_num, output_num);
}

__global__ void cu_train_weight(real input[], real grad[], real weight[], size_t input_num, size_t weight_num, real study_rate)
{
	size_t thread_num = gridDim.x * blockDim.x;
	size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
	while (thread_idx < weight_num)
	{
		size_t neuro_idx = thread_idx / input_num;
		size_t weight_idx = thread_idx % input_num;

		weight[thread_idx] += input[weight_idx] * grad[neuro_idx];// *study_rate;

		thread_idx += thread_num;
	}
}

__global__ void cu_train_bias(real grad[], real bias[], size_t output_num, real study_rate)
{
	size_t thread_num = gridDim.x * blockDim.x;
	size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
	while (thread_idx < output_num)
	{
		bias[thread_idx] += grad[thread_idx];// *study_rate;

		thread_idx += thread_num;
	}
}

__global__ void cu_target_to_grad(real output[], real target[], real grad[], real study_rate, size_t output_num)
{
	size_t thread_num = gridDim.x * blockDim.x;
	size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
	while (thread_idx < output_num)
	{
		grad[thread_idx] = (target[thread_idx] - output[thread_idx]) * study_rate;

		thread_idx += thread_num;
	}
}

__global__ void cu_grad_to_grad_merge(real gradin[], real weight[], real merge[], size_t input_num, size_t output_num)
{
	size_t thread_num = gridDim.x * blockDim.x;
	size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
	size_t weight_num = input_num * output_num;
	while (thread_idx < weight_num)
	{
		size_t neuro_idx = thread_idx / input_num;
		size_t weight_idx = thread_idx % input_num;
		merge[weight_idx * output_num + neuro_idx] = gradin[neuro_idx] * weight[thread_idx];

		thread_idx += thread_num;
	}
}

__global__ void cu_merge_to_grad(real merge[], real grad[], size_t input_num, size_t output_num)
{
	size_t thread_num = gridDim.x * blockDim.x;
	size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
	while (thread_idx < input_num)
	{
		grad[thread_idx] = merge[thread_idx * output_num];

		thread_idx += thread_num;
	}
}

void calculate_layer_grad(real output[], real target[], real grad[], real study_rate, size_t output_num)
{
	cu_target_to_grad<<<get_block_num(output_num), THREAD_NUM_PER_BLOCK >>>
		(output, target, grad, study_rate, output_num);
}

void calculate_layer_grad(real gradin[], real weight[], real grad[], size_t input_num, size_t output_num, real merge[])
{
	cu_grad_to_grad_merge<<<get_block_num(input_num * output_num), THREAD_NUM_PER_BLOCK >>>
		(gradin, weight, merge, input_num, output_num);
	calculate_merge(merge, output_num, input_num);
	cu_merge_to_grad<<<get_block_num(input_num), THREAD_NUM_PER_BLOCK >>>
		(merge, grad, input_num, output_num);
}

void calculate_layer_train(real input[], real grad[], real weight[], real bias[], size_t input_num, size_t output_num, real study_rate)
{
	cu_train_weight<<<get_block_num(input_num * output_num), THREAD_NUM_PER_BLOCK >>>
		(input, grad, weight, input_num, input_num * output_num, study_rate);
	cu_train_bias<<<get_block_num(output_num), THREAD_NUM_PER_BLOCK >>>
		(grad, bias, output_num, study_rate);
}

template<typename srctype, typename dsttype>
__global__ void cuda_translate_data_format(void *dst, const void *src, size_t pixelNum)
{
	size_t thread_num = gridDim.x * blockDim.x;
	size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
	while (thread_idx < pixelNum)
	{
		if ((thread_idx & 0x03) != 3)
		{
			((dsttype *)dst)[thread_idx] = (((dsttype)((const srctype *)src)[((thread_idx >> 2) * 3) + (thread_idx & 0x03)]) + (dsttype)1.0) * (dsttype)0.5;
		}
		else
		{
			((dsttype *)dst)[thread_idx] = (dsttype)0;
		}
		thread_idx += thread_num;
	}
}

void translate_data_format(void *dst, const void *src, size_t pixelNum, enum_translate_type type)
{
	pixelNum *= 4;
	switch (type)
	{
	case float_to_double:
		cuda_translate_data_format<float, double><<<get_block_num(pixelNum), THREAD_NUM_PER_BLOCK >>>(dst, src, pixelNum);
		break;
	case double_to_float:
		cuda_translate_data_format<double, float><<<get_block_num(pixelNum), THREAD_NUM_PER_BLOCK >>>(dst, src, pixelNum);
		break;
	case real_to_double:
		cuda_translate_data_format<real, double><<<get_block_num(pixelNum), THREAD_NUM_PER_BLOCK >>>(dst, src, pixelNum);
		break;
	case real_to_float:
		cuda_translate_data_format<real, float><<<get_block_num(pixelNum), THREAD_NUM_PER_BLOCK >>>(dst, src, pixelNum);
		break;
	case double_to_real:
		cuda_translate_data_format<double, real><<<get_block_num(pixelNum), THREAD_NUM_PER_BLOCK >>>(dst, src, pixelNum);
		break;
	case float_to_real:
		cuda_translate_data_format<float, real><<<get_block_num(pixelNum), THREAD_NUM_PER_BLOCK >>>(dst, src, pixelNum);
		break;
	default:
		break;
	}
}

__global__ void cuda_set_value(real data[], size_t count, real val)
{
	size_t thread_num = gridDim.x * blockDim.x;
	size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
	while (thread_idx < count)
	{
		data[thread_idx] = val;
		thread_idx += thread_num;
	}
}

void set_value(real data[], size_t count, real val)
{
	cuda_set_value<<<get_block_num(count), THREAD_NUM_PER_BLOCK >>> (data, count, val);
}

__device__ size_t cuda_get_rand_value(size_t seed)
{
	const static size_t a = 701507, b = 696497;
	char *p = (char *)&seed;
	size_t hash = a;
	hash ^= p[0];
	hash *= b;
	hash ^= p[1];
	hash *= b;
	hash ^= p[2];
	hash *= b;
	hash ^= p[3];
	hash *= b;
	return hash;
}

__global__ void cuda_set_rand_value(real data[], size_t count)
{
	size_t thread_num = gridDim.x * blockDim.x;
	size_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
	while (thread_idx < count)
	{
		data[thread_idx] = (real)((cuda_get_rand_value((size_t)&data[count]) & 0x0FFFFFF) - 0x07FFFFF) / (real)(0x07FFFFF);
		thread_idx += thread_num;
	}
}

void set_rand_value(real data[], size_t count)
{
	cuda_set_rand_value<<<get_block_num(count), THREAD_NUM_PER_BLOCK >>> (data, count);
}

/*#include <iostream>

int main()
{
	const static size_t in_num = 96 * 96 * 4;
	const static size_t out_num = 1024 * 2;
	const static size_t wei_num = in_num * out_num;

	cu_array<real> in(in_num), wei(wei_num), bias(out_num), tar(out_num);
	cu_array<real> out(out_num), grd(out_num), mrg(wei_num);
	
	for (size_t i = 0; i < in_num; ++i)
	{
		in[i] = ((real) rand() / (real) RAND_MAX);
	}
	in.set();
	for (size_t i = 0; i < wei_num; ++i)
	{
		wei[i] = ((real) rand() / (real) RAND_MAX);
	}
	wei.set();
	for (size_t i = 0; i < out_num; ++i)
	{
		bias[i] = ((real) rand() / (real) RAND_MAX);
		tar[i] = (real) i / (real) out_num;
	}
	bias.set();
	tar.set();

	calculate_layer_output(in(), wei(), bias(), out(), in_num, out_num, mrg());
	for (int i = 0; i < 1000000; ++i)
	{
		calculate_layer_grad(out(), tar(), grd(), out_num);
		calculate_layer_train(in(), grd(), wei(), bias(), in_num, out_num, 0.0001);
		calculate_layer_output(in(), wei(), bias(), out(), in_num, out_num, mrg());

		if ((i + 1) % 10 == 0)
		{
			out.get();
			for (int j = 0; j < out_num; ++j)
			{
				printf(" %15f", tar[j] - out[j]);
			}
			printf("\n\n");
		}
	}

	system("pause");
	return 0;
}*/
