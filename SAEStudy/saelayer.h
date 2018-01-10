#ifndef SAELAYER_H
#define SAELAYER_H

#include "libcu.h"

class SAELayer
{
public:
    SAELayer(size_t inputNum, size_t outputNum);

public:
    real *encode(real *input);
    real *decode(real *output);
    void train(real *input, real studyRate);

public:
    real *getEncode();
    real *getDecode();

private:
    size_t m_InNum;
    cu_array<real> m_EnWei, m_EnBias, m_EnOut, m_EnGrad;
    size_t m_EnNum;
    cu_array<real> m_DeWei, m_DeBias, m_DeOut, m_DeGrad;
    size_t m_DeNum;
    cu_array<real> m_Merge;
};

#endif // SAELAYER_H
