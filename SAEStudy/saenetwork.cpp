#include "saenetwork.h"

#include "config.h"

SAENetwork::SAENetwork(size_t inputNum, size_t layerNum, const size_t *neuroNum) :
    m_ppLayers(new SAELayer*[layerNum]), m_LayNum(layerNum)
{
    for (size_t layIdx = 0; layIdx < layerNum; ++layIdx)
    {
        size_t outputNum = neuroNum[layIdx];
        m_ppLayers[layIdx] = new SAELayer(inputNum, outputNum);
        inputNum = outputNum;
    }
}

SAENetwork::~SAENetwork()
{
    for (size_t layIdx = 0; layIdx < m_LayNum; ++layIdx)
    {
        delete m_ppLayers[layIdx];
    }
    delete[] m_ppLayers;
}

real *SAENetwork::encode(real *input)
{
    for (size_t layIdx = 0; layIdx < m_LayNum; ++layIdx)
    {
        input = m_ppLayers[layIdx]->encode(input);
    }
    return input;
}

real *SAENetwork::decode(real *output)
{
    size_t layIdx = m_LayNum;
    while(layIdx)
    {
        output = m_ppLayers[--layIdx]->decode(output);
    }
    return output;
}

void SAENetwork::train(real *input, real studyRate)
{
    for (size_t layIdx = 0; layIdx < m_LayNum; ++layIdx)
    {
        m_ppLayers[layIdx]->train(input, studyRate);

        input = m_ppLayers[layIdx]->getEncode();
        studyRate *= STUDY_RATE_DECREASE_RATE;
    }
}
