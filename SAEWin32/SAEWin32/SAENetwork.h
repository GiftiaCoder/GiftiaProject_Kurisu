#pragma once

#include "SAELayer.h"

class CSAENetwork
{
public:
	CSAENetwork(size_t inputNum, size_t layerNum, const size_t *neuroNum);
	~CSAENetwork();

public:
	real *Encode(real *input);
	real *Decode(real *output);
	void Train(real *input, real studyRate);

private:
	size_t m_LayNum;
	CSAELayer *m_pLayers;
};

