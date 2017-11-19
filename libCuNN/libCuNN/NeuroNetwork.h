
#ifndef _NEURO_NETWORK_H_
#define _NEURO_NETWORK_H_

#ifndef _CUNN_BASE_TYPE_H_
#define _CUNN_BASE_TYPE_H_
typedef float real;
typedef unsigned int count;
#endif

class NeuroNetwork
{
public:
	virtual void Output(real input[]) = 0;
	virtual void Output(real input[], count from_lay, count to_lay) = 0;
	
	virtual void Train(real input[], real target[], real study_rate) = 0;
	virtual void Train(real input[], real target[], count from_lay, count to_lay, real study_rate) = 0;

public:
	virtual real *GetOutput() = 0;
	virtual real *GetOutput(count lay) = 0;
};

#endif
