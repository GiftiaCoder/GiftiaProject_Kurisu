
#include "BPNetwork.h"

BPNetwork::BPNetwork() {}

BPNetwork::BPNetwork(count input_num, count lay_num, count neuro_num[]) {
	Init(input_num, lay_num, neuro_num);
}

BPNetwork::~BPNetwork() {
	delete[] m_pLayers;
}

void BPNetwork::Init(count input_num, count lay_num, count neuro_num[]) {
	m_pLayers = new NeuroLayer[lay_num];
	m_GradTop = neuro_num[lay_num - 1];
	m_LayerNum = lay_num;
	count output_num;
	for (count layIdx = 0; layIdx < lay_num; ++layIdx) {
		m_pLayers[layIdx].Init(input_num, output_num = neuro_num[layIdx]);
		input_num = output_num;
	}
}

void BPNetwork::Output(real input[]) {
	Output(input, 0, m_LayerNum);
}

void BPNetwork::Output(real input[], count from_lay, count to_lay) {
	NeuroLayer *pBegin = m_pLayers + from_lay, *pEnd = m_pLayers + to_lay;
	while (pBegin != pEnd) {
		pBegin->Output(input);
		input = pBegin->GetOutput();
		++pBegin;
	}
}

void BPNetwork::Train(real input[], real target[], real study_rate) {
	Train(input, target, 0, m_LayerNum, study_rate);
}

void BPNetwork::Train(real input[], real target[], count from_lay, count to_lay, real study_rate) {
	NeuroLayer *pBegin = m_pLayers + from_lay, *pEnd = m_pLayers + to_lay;
	NeuroLayer *pLay = pBegin;
	while (pLay != pEnd) {
		pLay->SetInput(input);
		input = pLay->GetOutput();
		++pLay;
	}

	float *grad = m_GradTop.GetData();
	(pEnd - 1)->Grad(target, grad);
	while (pLay != pBegin) {
		--pLay;
		pLay->DoTrain(grad, study_rate);
		grad = pLay->GetGrad();
	}
}

real *BPNetwork::GetOutput() {
	return GetOutput(m_LayerNum - 1);
}

real *BPNetwork::GetOutput(count lay) {
	return (m_pLayers + lay)->GetOutput();
}
