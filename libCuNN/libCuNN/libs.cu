
#include "libs.cuh"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

__global__ void cudaMultiply(real input[], real weight[], real merge[], count input_num, count weight_num) {
	count thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	count thread_num = gridDim.x * blockDim.x;
	while (thread_id < weight_num) {
		merge[thread_id] = weight[thread_id] * input[thread_id % input_num];
		thread_id += thread_num;
	}
}

__global__ void cudaMergePlus(real merge[], count input_num, count output_num, count base_num, count group_num, count merge_num) {
	count thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	count thread_num = gridDim.x * blockDim.x;

	while (thread_id < merge_num) {
		count merge_idx = (thread_id / group_num) * input_num + (thread_id % group_num);
		merge[merge_idx] += merge[merge_idx + base_num];
		thread_id += thread_num;
	}
}

__global__ void cudaGetOutput(real merge[], real bias[], real output[], count input_num, count output_num) {
	count thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	count thread_num = gridDim.x * blockDim.x;
	while (thread_id < output_num) {
		output[thread_id] = tanh(merge[thread_id * input_num] + bias[thread_id]);
		thread_id += thread_num;
	}
}

void MergePlus(real merge[], count input_num, count output_num) {
	count group_num = input_num >> 1;
	count base_num = group_num + (input_num & 0x1);

	while (group_num > 0) {
		count merge_num = group_num * output_num;
		cudaMergePlus << <cudaGetGD(merge_num), cudaGetBD(merge_num) >> >(merge, input_num, output_num, base_num, group_num, merge_num);

		group_num = base_num >> 1;
		base_num = group_num + (base_num & 0x1);
	}
}

void GetLayerOutput(real input[], real bias[], real weight[], real output[], count input_num, count output_num, real merge[]) {
	count weight_num = input_num * output_num;
	count gd;
	count bd;

	cudaMultiply << <cudaGetGD(weight_num), cudaGetBD(weight_num) >> >(input, weight, merge, input_num, weight_num);
	MergePlus(merge, input_num, output_num);
	cudaGetOutput << <cudaGetGD(output_num), cudaGetBD(output_num) >> >(merge, bias, output, input_num, output_num);
}

__global__ void cudaGetGradTop(real output[], real target[], real grad[], count output_num) {
	count thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	count thread_num = gridDim.x * blockDim.x;

	while (thread_id < output_num) {
		grad[thread_id] = target[thread_id] - output[thread_id];
		thread_id += thread_num;
	}
}

__global__ void cudaGetGrad(real gradout[], real weight[], real merge[], count input_num, count output_num, count weight_num) {
	count thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	count thread_num = gridDim.x * blockDim.x;

	while (thread_id < weight_num) {
		count neuro_idx = thread_id / input_num;
		count input_idx = thread_id % input_num;
		merge[output_num * input_idx + neuro_idx] = gradout[neuro_idx] * weight[thread_id];
		thread_id += thread_num;
	}
}

__global__ void cudaTrainWeight(real input[], real grad[], real weight[], count input_num, count output_num, count weight_num, real study_rate) {
	count thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	count thread_num = gridDim.x * blockDim.x;

	while (thread_id < weight_num) {
		count neuro_idx = thread_id / input_num;
		count input_idx = thread_id % input_num;
		weight[thread_id] += input[input_idx] * grad[neuro_idx] * study_rate;
		thread_id += thread_num;
	}
}

__global__ void cudaTrainBias(real grad[], real bias[], count output_num, real study_rate) {
	count thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	count thread_num = gridDim.x * blockDim.x;

	while (thread_id < output_num) {
		bias[thread_id] += grad[thread_id] * study_rate;
		thread_id += thread_num;
	}
}

__global__ void cudaCopyMergeToGrad(real merge[], real grad[], count input_num, count output_num) {
	count thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	count thread_num = gridDim.x * blockDim.x;

	while (thread_id < output_num) {
		grad[thread_id] = merge[thread_id * input_num];
		thread_id += thread_num;
	}
}

void TrainTopLayer(real input[], real output[], real target[], real weight[], real bias[], count input_num, count output_num, real gradin[], real gradout[], real merge[], real study_rate) {
	count weight_num = input_num * output_num;

	cudaGetGradTop<<<cudaGetGD(output_num), cudaGetBD(output_num) >> >(output, target, gradin, output_num);
	cudaTrainWeight<<<cudaGetGD(weight_num), cudaGetBD(weight_num) >> >(input, gradin, weight, input_num, output_num, weight_num, study_rate);
	cudaTrainBias<<<cudaGetGD(output_num), cudaGetBD(output_num) >> >(gradin, bias, output_num, study_rate);

	cudaGetGrad << <cudaGetGD(weight_num), cudaGetBD(weight_num) >> >(gradin, weight, merge, input_num, output_num, weight_num);
	MergePlus(merge, input_num, output_num);
	cudaCopyMergeToGrad << <cudaGetGD(output_num), cudaGetBD(output_num) >> >(merge, gradout, input_num, output_num);
}

void TrainHiddenLayer(real input[], real weight[], real gradin[], real bias[], count input_num, count output_num, real gradout[], real merge[], real study_rate) {
	count weight_num = input_num * output_num;

	cudaTrainWeight << <cudaGetGD(weight_num), cudaGetBD(weight_num) >> >(input, gradin, weight, input_num, output_num, weight_num, study_rate);
	cudaTrainBias << <cudaGetGD(output_num), cudaGetBD(output_num) >> >(gradin, bias, output_num, study_rate);

	cudaGetGrad << <cudaGetGD(weight_num), cudaGetBD(weight_num) >> >(gradin, weight, merge, input_num, output_num, weight_num);
	MergePlus(merge, input_num, output_num);
	cudaCopyMergeToGrad << <cudaGetGD(output_num), cudaGetBD(output_num) >> >(merge, gradout, input_num, output_num);
}

