#include "Network.h"


CNetwork::CNetwork(count lnum, count nnum[], count inum) :
	m_pLayer(new CLayer[lnum]),
	m_GradTop(nnum[lnum - 1]),
	m_LayerNum(lnum) {
	for (count lidx = 0; lidx < lnum; ++lidx) {
		m_pLayer[lidx].Init(inum, nnum[lidx]);
		inum = nnum[lidx];
	}
}

CNetwork::~CNetwork() {
	delete[] m_pLayer;
}

real* CNetwork::Output(real input[]) {
	for (count lidx = 0; lidx < m_LayerNum; ++lidx) {
		m_pLayer[lidx].CalculateOutput(input);
		input = m_pLayer[lidx].GetOutput();
	}
	return input;
}

void CNetwork::Train(real input[], real target[], real study_rate) {
	m_pLayer[m_LayerNum - 1].TrainWithTarget(input, target, m_GradTop.GetData(), study_rate);
	for (int lidx = m_LayerNum - 2; lidx >= 0; --lidx) {
		m_pLayer[lidx].TrainWithGrad(input, m_pLayer[lidx + 1].GetGrad(), study_rate);
	}
}

real* CNetwork::Output(real input[], count begLayIdx, count endLayIdx) {

}

void Train(real input[], real target[], count begLayIdx, count endLayIdx, real study_rate) {

}
