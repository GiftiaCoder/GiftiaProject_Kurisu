
#include "libs.cuh"
#include "util.cuh"

#include "Network.h"

#define INPUT_NUM (32)
#define HIDDEN_NUM (24)
#define OUTPUT_NUM (14)

#define TRAIN_LOOP_TIME (1024)

int main() {
	count nnum[] = {
		HIDDEN_NUM, OUTPUT_NUM, 
	};

	CCudaMemory<real> cinput(INPUT_NUM), ctarget(OUTPUT_NUM);
	CHostMemory<real> hinput(INPUT_NUM), htarget(OUTPUT_NUM), houtput(OUTPUT_NUM);

	CNetwork network(2, nnum, INPUT_NUM);

	real in = 0.0F;
	for (count i = 0; i < INPUT_NUM; ++i) {
		hinput[i] = (in += 0.02F);
	}
	real out = 0.0F;
	for (count i = 0; i < OUTPUT_NUM; ++i) {
		htarget[i] = (out += 0.02F);
	}
	cinput.CopyFrom(hinput);
	ctarget.CopyFrom(htarget);

	for (count i = 0; i < OUTPUT_NUM; ++i) {
		printf("%15f ", htarget[i]);;
	}
	printf("\n\n");

	for (count t = 0; t < TRAIN_LOOP_TIME; ++t) {
		real *output = network.Output(cinput.GetData());

		houtput.CopyFromCuda(output);
		for (count i = 0; i < OUTPUT_NUM; ++i) {
			printf("%15f ", houtput[i]);
		}
		printf("\n");

		network.Train(cinput.GetData(), ctarget.GetData(), 0.01F);
	}

	system("pause");
	return 0;
}
