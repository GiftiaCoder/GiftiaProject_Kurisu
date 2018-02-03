#include "SAELayer.h"

CSAELayer::CSAELayer(size_t inputNum, size_t outputNum) :
	m_InNum(inputNum),
	m_EnWei(inputNum * outputNum, true), m_EnBias(outputNum, true), m_EnOut(outputNum), m_EnGrad(outputNum),
	m_EnNum(outputNum),
	m_DeWei(outputNum * inputNum, true), m_DeBias(inputNum, true), m_DeOut(inputNum), m_DeGrad(inputNum), 
	m_DeNum(inputNum), 
	m_Merge(inputNum * outputNum)
{
	int weightNum = inputNum * outputNum;
	for (int i = 0; i < weightNum; ++i)
	{
		m_EnWei[i] = (((real)(rand()) / (real)RAND_MAX) - (real)0.5) * (real)(2.0 * 2.0);
		m_DeWei[i] = (((real)(rand()) / (real)RAND_MAX) - (real)0.5) * (real)(2.0 * 2.0);
	}
	m_EnWei.set();
	m_DeWei.set();

	for (int i = 0; i < outputNum; ++i)
	{
		m_EnBias[i] = (((real)(rand()) / (real)RAND_MAX) - (real)0.5) * (real)(2.0 * 2.0);
	}
	m_EnBias.set();

	for (int i = 0; i < inputNum; ++i)
	{
		m_DeBias[i] = (((real)(rand()) / (real)RAND_MAX) - (real)0.5) * (real)(2.0 * 2.0);
	}
	m_DeBias.set();

	set_value(m_EnOut(), m_EnOut.size(), 0);
	set_value(m_EnGrad(), m_EnGrad.size(), 0);
	set_value(m_DeOut(), m_DeOut.size(), 0);
	set_value(m_DeGrad(), m_DeGrad.size(), 0);
	set_value(m_Merge(), m_Merge.size(), 0);
}

real *CSAELayer::Encode(real *input)
{
	calculate_layer_output(input, m_EnWei(), m_EnBias(), m_EnOut(), m_InNum, m_EnNum, m_Merge());
	return m_EnOut();
}

real *CSAELayer::Decode(real *output)
{
	calculate_layer_output(output, m_DeWei(), m_DeBias(), m_DeOut(), m_EnNum, m_DeNum, m_Merge());
	return m_DeOut();
}

void CSAELayer::Train(real *input, real studyRate)
{
	real *gen = Decode(Encode(input));

	calculate_layer_grad(gen, input, m_DeGrad(), studyRate, m_DeNum);
	calculate_layer_train(m_EnOut(), m_DeGrad(), m_DeWei(), m_DeBias(), m_EnNum, m_DeNum, studyRate);

	calculate_layer_grad(m_DeGrad(), m_EnWei(), m_EnGrad(), m_InNum, m_EnNum, m_Merge());
	calculate_layer_train(input, m_EnGrad(), m_EnWei(), m_EnBias(), m_InNum, m_EnNum, studyRate);
}

real *CSAELayer::GetEncode()
{
	return m_EnOut();
}

real *CSAELayer::GetDecode()
{
	return m_DeOut();
}
