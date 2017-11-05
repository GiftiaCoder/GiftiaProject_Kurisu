#include "Layer.h"

CLayer::CLayer() {}

CLayer::CLayer(count input_num, count output_num) {
	Init(input_num, output_num);
}

void CLayer::Init(count input_num, count output_num) {
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

void CLayer::CalculateOutput(real input[]) {
	GetLayerOutput(input, 
		m_Bias.GetData(), m_Weight.GetData(), m_Output.GetData(), 
		m_InputNum, m_OutputNum, m_Merge.GetData());
}

void CLayer::TrainWithTarget(real input[], real target[], real gradin[], real study_rate) {
	TrainTopLayer(input, 
		m_Output.GetData(), target, 
		m_Weight.GetData(), m_Bias.GetData(), 
		m_InputNum, m_OutputNum, 
		gradin, m_Grad.GetData(), m_Merge.GetData(), 
		study_rate);
}

void CLayer::TrainWithGrad(real input[], real grad[], real study_rate) {
	TrainHiddenLayer(input, 
		m_Weight.GetData(), grad, m_Bias.GetData(), 
		m_InputNum, m_OutputNum, m_Grad.GetData(),
		m_Merge.GetData(), 
		study_rate);
}

real* CLayer::GetOutput() {
	return m_Output.GetData();
}

real* CLayer::GetGrad() {
	return m_Grad.GetData();
}
