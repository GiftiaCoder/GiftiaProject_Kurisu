#include "SAENetwork.h"

#include "Config.h"

CSAENetwork::CSAENetwork(size_t inputNum, size_t layerNum, const size_t *neuroNum) :
	m_pLayers((CSAELayer *)new char[layerNum * sizeof(CSAELayer)]), m_LayNum(layerNum)
{
	for (size_t layIdx = 0; layIdx < layerNum; ++layIdx)
	{
		size_t outputNum = neuroNum[layIdx];
		m_pLayers[layIdx].CSAELayer::CSAELayer(inputNum, outputNum); // explicitly calling the constructor
		inputNum = outputNum;
	}
}

CSAENetwork::~CSAENetwork()
{
	delete[] m_pLayers;
}

real *CSAENetwork::Encode(real *input)
{
	for (size_t layIdx = 0; layIdx < m_LayNum; ++layIdx)
	{
		input = m_pLayers[layIdx].Encode(input);
	}
	return input;
}

real *CSAENetwork::Decode(real *output)
{
	size_t layIdx = m_LayNum;
	while (layIdx)
	{
		output = m_pLayers[--layIdx].Decode(output);
	}
	return output;
}

void CSAENetwork::Train(real *input, real studyRate)
{
	for (size_t layIdx = 0; layIdx < m_LayNum; ++layIdx)
	{
		m_pLayers[layIdx].Train(input, studyRate);

		input = m_pLayers[layIdx].GetEncode();
		//studyRate *= CConfig::STUDY_RATE_DECREASE_RATE;
	}
}
