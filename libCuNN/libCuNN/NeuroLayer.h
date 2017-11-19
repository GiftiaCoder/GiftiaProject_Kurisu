#pragma once

#include "init.cuh"
#include "libs.cuh"

class NeuroLayer
{
private:
	CCudaMemory<real> m_Bias, m_Weight, m_Output, m_Grad, m_Merge;
	count m_InputNum, m_OutputNum;

public:
	NeuroLayer();
	NeuroLayer(count input_num, count output_num);

	void Init(count input_num, count output_num);

public:
	void Output(real input[]);

public:
	real* GetOutput();
	count GetInputNum();
	count GetOutputNum();
	real* GetGrad();

public:
	void Grad(real target[], real grad[]);
	void SetInput(real input[]);
	void DoTrain(real grad[], real study_rate);

private:
	real *m_TmpTrainInput;
};

