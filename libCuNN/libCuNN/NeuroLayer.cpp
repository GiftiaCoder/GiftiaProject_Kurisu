
#include "NeuroLayer.h"

NeuroLayer::NeuroLayer() {}

NeuroLayer::NeuroLayer(count input_num, count output_num) {
	Init(input_num, output_num);
}

void NeuroLayer::Init(count input_num, count output_num) {
	count weight_num = input_num * output_num;
	
	m_Bias.Init(output_num);
	m_Bias.SetRand(1.0, -1.0);

	m_Weight.Init(weight_num);
	m_Weight.SetRand(1.0, -1.0);

	m_Output.Init(output_num);
	m_Merge.Init(weight_num);
	m_Grad.Init(input_num);
	
	m_InputNum = input_num;
	m_OutputNum = output_num;
}

void NeuroLayer::Output(real input[]) {
	GetLayerOutput(input, 
		m_Bias.GetData(), m_Weight.GetData(), m_Output.GetData(), 
		m_InputNum, m_OutputNum, m_Merge.GetData());
}

real* NeuroLayer::GetOutput() {
	return m_Output.GetData();
}

count NeuroLayer::GetOutputNum() {
	return m_OutputNum;
}

count NeuroLayer::GetInputNum() {
	return m_InputNum;
}

real* NeuroLayer::GetGrad() {
	return m_Grad.GetData();
}

void NeuroLayer::Grad(real target[], real grad[]) {
	GetTopGrad(m_Output.GetData(), target, grad, m_OutputNum);
}

void NeuroLayer::SetInput(real input[]) {
	m_TmpTrainInput = input;
}

void NeuroLayer::DoTrain(real grad[], real study_rate) {
	TrainHiddenLayer(m_TmpTrainInput,
		m_Weight.GetData(), grad, m_Bias.GetData(),
		m_InputNum, m_OutputNum, m_Grad.GetData(),
		m_Merge.GetData(),
		study_rate);
}
