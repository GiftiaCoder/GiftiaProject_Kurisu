#pragma once

#include "libcu.h"

class CSAELayer
{
public:
	CSAELayer(size_t inputNum, size_t outputNum);

public:
	real *Encode(real *input);
	real *Decode(real *output);
	void Train(real *input, real studyRate);

public:
	real *GetEncode();
	real *GetDecode();

private:
	size_t m_InNum;

	cu_array<real> m_EnWei, m_EnBias, m_EnOut, m_EnGrad;
	size_t m_EnNum;

	cu_array<real> m_DeWei, m_DeBias, m_DeOut, m_DeGrad;
	size_t m_DeNum;

	cu_array<real> m_Merge;
};

