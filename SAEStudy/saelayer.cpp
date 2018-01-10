#include "saelayer.h"

SAELayer::SAELayer(size_t inputNum, size_t outputNum) :
    m_InNum(inputNum),
    m_EnWei(inputNum * outputNum), m_EnBias(outputNum), m_EnOut(outputNum), m_EnGrad(outputNum),
    m_EnNum(outputNum),
    m_DeWei(inputNum * outputNum), m_DeBias(inputNum), m_DeOut(inputNum), m_DeGrad(inputNum),
    m_DeNum(inputNum),
    m_Merge(inputNum * outputNum)
{
    // do nothing
}

real *SAELayer::encode(real *input)
{
    calculate_layer_output(input, m_EnWei(), m_EnBias(), m_EnOut(), m_InNum, m_EnNum, m_Merge());
    return m_EnOut();
}

real *SAELayer::decode(real *output)
{
    calculate_layer_output(output, m_DeWei(), m_DeBias(), m_DeOut(), m_EnNum, m_DeNum, m_Merge());
    return m_DeOut();
}

void SAELayer::train(real *input, real studyRate)
{
    real *gen = decode(encode(input));

    calculate_layer_grad(gen, input, m_DeGrad(), m_DeNum);
    calculate_layer_train(m_EnOut(), m_DeGrad(), m_DeWei(), m_DeBias(), m_EnNum, m_DeNum, studyRate);

    calculate_layer_grad(m_DeGrad(), m_EnWei(), m_EnGrad(), m_InNum, m_EnNum, m_Merge());
    calculate_layer_train(input, m_EnGrad(), m_EnWei(), m_EnBias(), m_InNum, m_EnNum, studyRate);
}

real *SAELayer::getEncode()
{
    return m_EnOut();
}

real *SAELayer::getDecode()
{
    return m_DeOut();
}
