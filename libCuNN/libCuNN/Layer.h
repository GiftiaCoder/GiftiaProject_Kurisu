#pragma once

#include "init.cuh"
#include "libs.cuh"

class CLayer
{
private:
	CCudaMemory<real> m_Bias, m_Weight, m_Output, m_Grad, m_Merge;
	count m_InputNum, m_OutputNum;

public:
	CLayer();
	CLayer(count input_num, count output_num);

	void Init(count input_num, count output_num);

public:
	void CalculateOutput(real input[]);
	void TrainWithTarget(real input[], real target[], real gradin[], real study_rate);
	void TrainWithGrad(real input[], real grad[], real study_rate);

public:
	real* GetOutput();
	real* GetGrad();
};

