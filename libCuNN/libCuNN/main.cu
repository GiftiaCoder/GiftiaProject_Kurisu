
#include "libs.cuh"
#include "util.cuh"

#include "Network.h"

#define INPUT_NUM (16)
#define HIDDEN_NUM (8)
#define OUTPUT_NUM (4)

int main() {
	count nnum[] = {
		HIDDEN_NUM, OUTPUT_NUM, 
	};

	CCudaMemory<real> cinput(16), ctarget(4);
	CHostMemory<real> hinput(16), htarget(4), houtput(4);

	printf("%d\n", sizeof(cinput));

	CNetwork network(2, nnum, INPUT_NUM);

	real in = 0.05F;
	for (count i = 0; i < INPUT_NUM; ++i) {
		hinput[i] = (in += 0.05F);
	}
	real out = 0.1F;
	for (count i = 0; i < OUTPUT_NUM; ++i) {
		htarget[i] = (out += 0.1F);
	}
	cinput.CopyFrom(hinput);
	ctarget.CopyFrom(htarget);

	for (count i = 0; i < OUTPUT_NUM; ++i) {
		printf("%15f ", htarget[i]);;
	}
	printf("\n\n");

	for (count t = 0; t < 16; ++t) {
		real *output = network.Output(cinput.GetData());

		houtput.CopyFromCuda(output);
		for (count i = 0; i < OUTPUT_NUM; ++i) {
			printf("%15f ", houtput[i]);
		}
		printf("\n");

		network.Train(cinput.GetData(), ctarget.GetData(), 1.0F);
	}

	system("pause");
	return 0;
}
