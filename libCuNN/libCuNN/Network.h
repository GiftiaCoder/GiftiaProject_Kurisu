#pragma once

#include "Layer.h"

class CNetwork
{
private:
	CLayer *m_pLayer;
	CCudaMemory<real> m_GradTop;
	count m_LayerNum;

public:
	CNetwork(count lnum, count nnum[], count inum);
	~CNetwork();

public:
	real* Output(real input[]);
	void Train(real input[], real target[], real study_rate);
};

