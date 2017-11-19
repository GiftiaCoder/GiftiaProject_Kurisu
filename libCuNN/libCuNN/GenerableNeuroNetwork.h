
#ifndef _GENERABLE_NEURO_NETWORK_H_
#define _GENERABLE_NEURO_NETWORK_H_

#include "NeuroNetwork.h"

class GenerableNeuroNetwork : public NeuroNetwork {
public:
	virtual void Generate(real output[]) = 0;
	virtual void Generate(real output[], count from_lay, count to_aly) = 0;

public:
	virtual real *GetGenerate() = 0;
	virtual real *GetGenerate(count lay) = 0;
};

#endif
