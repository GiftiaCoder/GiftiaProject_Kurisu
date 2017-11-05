
#ifndef _LIBS_CUH_
#define _LIBS_CUH_

#include "util.cuh"

void GetLayerOutput(real input[], real bias[], real weight[], real output[], count input_num, count output_num, real merge[]);

void TrainTopLayer(real input[], real output[], real target[], real weight[], real bias[], count input_num, count output_num, real gradin[], real gradout[], real merge[], real study_rate);

void TrainHiddenLayer(real input[], real weight[], real gradout[], real bias[], count input_num, count output_num, real grad[], real merge[], real study_rate);

#endif
