#include "AENetwork.h"

AENetwork::AENetwork(count input_num, count lay_num, count neuro_num[]) : 
	m_pLayers(new BPNetwork[lay_num]), 
	m_LayerNum(lay_num) {
	const static count LAYER_NETWORK_LAY_NUM = 2;
	count lay_network_neuro_num[LAYER_NETWORK_LAY_NUM];
	
	for (count layIdx = 0; layIdx < lay_num; ++layIdx) {
		count output_num = neuro_num[layIdx];
		lay_network_neuro_num[0] = output_num;
		lay_network_neuro_num[1] = input_num;
		m_pLayers[layIdx].Init(input_num, 2, lay_network_neuro_num);
		input_num = output_num;
	}
}

AENetwork::~AENetwork() {
	delete[] m_pLayers;
}

void AENetwork::Output(real input[]) {
	Output(input, 0, m_LayerNum);
}

void AENetwork::Output(real input[], count from_lay, count to_lay) {
	BPNetwork *pLay = m_pLayers + from_lay, *pEnd = m_pLayers + to_lay;
	while (pLay != pEnd) {
		pLay->Output(input);
		input = pLay->GetOutput(0);
	}
}

void AENetwork::Train(real input[], real target[], real study_rate) {
	Train(input, target, 0, m_LayerNum, study_rate);
}

void AENetwork::Train(real input[], real target[], count from_lay, count to_lay, real study_rate) {
	BPNetwork *pLay = m_pLayers + from_lay, *pEnd = m_pLayers + to_lay;
	while (pLay != pEnd) {
		pLay->Train(input, input, study_rate);
		input = pLay->GetOutput(0);
	}
}

real *AENetwork::GetOutput() {
	return GetOutput(m_LayerNum - 1);
}

real *AENetwork::GetOutput(count lay) {
	return m_pLayers[lay].GetOutput(0);
}

void AENetwork::Generate(real output[]) {
	Generate(output, 0, m_LayerNum);
}

void AENetwork::Generate(real output[], count from_lay, count to_lay) {
	BPNetwork *pLay = m_pLayers + to_lay, *pBeg = m_pLayers;
	while (pLay != pBeg) {
		--pLay;
		pLay->Output(output, 1, 2);
		output = pLay->GetOutput();
	}
}

real *AENetwork::GetGenerate() {
	return GetGenerate(m_LayerNum - 1);
}

real *AENetwork::GetGenerate(count lay) {
	return m_pLayers[lay].GetOutput();
}
