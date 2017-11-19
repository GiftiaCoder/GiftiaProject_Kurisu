
#ifndef _LIB_CUNN_H_
#define _LIB_CUNN_H_

#include "AENetwork.h"
#include "BPNetwork.h"

#pragma comment(lib, "libCuNN.lib")

extern "C" {

	__declspec(dllimport) NeuroNetwork *CreateBPNetwork(count input_num, count lay_num, count neuro_num[]);

	__declspec(dllimport) GenerableNeuroNetwork *CreateAENetwork(count input_num, count lay_num, count neuro_num[]);

	__declspec(dllimport) void ReleaseNetwork(NeuroNetwork *network);

}

#endif

