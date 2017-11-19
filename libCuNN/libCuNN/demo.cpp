
#include <iostream>
#include "BPNetwork.h"

#define H0 20
#define H1 15
#define H2 13
#define H3 10

int main() {
	count neuroNum[3] = { H1, H2, H3 };
	BPNetwork network(H0, 3, neuroNum);

	CCudaMemory<real> cInput(H0), cTarget(H3);
	CHostMemory<real> hInput(H0), hOutput(H3), hTarget(H3);

	for (int i = 0; i < H0; ++i)
		hInput[i] = ((real)i) / ((float)H0);
	for (int i = 0; i < H3; ++i)
		hTarget[i] = ((real)i) / ((float)H3);

	cInput.CopyFrom(hInput);
	cTarget.CopyFrom(hTarget);
	
	for (int i = 0; i < 10000; ++i) {
		network.Output(cInput.GetData());
		network.Train(cInput.GetData(), cTarget.GetData(), 0.1);

		hOutput.CopyFromCuda(network.GetOutput());
		for (int j = 0; j < H3; ++j) {
			printf("%15f ", hOutput[j]);
		}
		printf("\n");
	}

	system("pause");
	return 0;
}
