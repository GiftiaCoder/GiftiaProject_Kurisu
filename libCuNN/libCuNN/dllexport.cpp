
#include <Windows.h>

#include "AENetwork.h"
#include "BPNetwork.h"

extern "C" {

	__declspec(dllexport) NeuroNetwork *CreateBPNetwork(count input_num, count lay_num, count neuro_num[]) {
		return new BPNetwork(input_num, lay_num, neuro_num);
	}

	__declspec(dllexport) GenerableNeuroNetwork *CreateAENetwork(count input_num, count lay_num, count neuro_num[]) {
		return new AENetwork(input_num, lay_num, neuro_num);
	}

	__declspec(dllexport) void ReleaseNetwork(NeuroNetwork *network) {
		delete[] network;
	}

}
