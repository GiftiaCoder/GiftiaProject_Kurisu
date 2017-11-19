
#ifndef _AENETWORK_H_
#define _AENETWORK_H_

#include "GenerableNeuroNetwork.h"
#include "BPNetwork.h"

class AENetwork : public GenerableNeuroNetwork
{
public:
	AENetwork(count input_num, count lay_num, count neuro_num[]);
	virtual ~AENetwork();

public:
	virtual void Output(real input[]) override;
	virtual void Output(real input[], count from_lay, count to_lay) override;

	virtual void Train(real input[], real target[], real study_rate) override;
	virtual void Train(real input[], real target[], count from_lay, count to_lay, real study_rate) override;

public:
	virtual real *GetOutput() override;
	virtual real *GetOutput(count lay) override;

public:
	virtual void Generate(real output[]) override;
	virtual void Generate(real output[], count from_lay, count to_aly) override;

public:
	virtual real *GetGenerate() override;
	virtual real *GetGenerate(count lay) override;

private:
	count m_LayerNum;
	BPNetwork *m_pLayers;
};

#endif
