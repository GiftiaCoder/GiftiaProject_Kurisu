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
<<<<<<< HEAD

	//real* Output(real input[], count begLayIdx, count endLayIdx);
	//void Train(real input[], real target[], count begLayIdx, count endLayIdx, real study_rate);
=======
>>>>>>> 6fe87d135d9f6e612352c9f7173863f6b5e786d0
};

