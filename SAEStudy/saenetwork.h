#ifndef SAENETWORK_H
#define SAENETWORK_H

#include "saelayer.h"

class SAENetwork
{
public:
    SAENetwork(size_t inputNum, size_t layerNum, const size_t *neuroNum);
    ~SAENetwork();

public:
    real *encode(real *input);
    real *decode(real *output);
    void train(real *input, real studyRate);

public:
    // TODO

private:
    size_t m_LayNum;
    SAELayer **m_ppLayers;
};

#endif // SAENETWORK_H
