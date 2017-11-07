#pragma once

#include "Layer.h"

class CAudoEncodeLayer {
public:
    CAudoEncodeLayer(count inum, count onum);
    
public:
    real* Output(real input[]);
    void Train(real input[], real study_rate);
};
