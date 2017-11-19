
#ifndef _BP_NETWORK_H_
#define _BP_NETWORK_H_

#include "NeuroNetwork.h"

#include "NeuroLayer.h"

class BPNetwork : public NeuroNetwork
{
public:
	BPNetwork();
	BPNetwork(count input_num, count lay_num, count neuro_num[]);
	virtual ~BPNetwork();

public:
	void Init(count input_num, count lay_num, count neuro_num[]);

public:
	virtual void Output(real input[]) override;
	virtual void Output(real input[], count from_lay, count to_lay) override;
	
	virtual void Train(real input[], real target[], real study_rate) override;
	virtual void Train(real input[], real target[], count from_lay, count to_lay, real study_rate) override;

public:
	virtual real *GetOutput() override;
	virtual real *GetOutput(count lay) override;
	
private:
	NeuroLayer *m_pLayers;
	CCudaMemory<real> m_GradTop;
	count m_LayerNum;
};

#endif
