#include "Config.h"

#include "libcu.h"

const size_t CConfig::SAENETWORK_INPUT_PIXEL_NUM = 96 * 96;
const size_t CConfig::SAENETWORK_INPUT_NUM = SAENETWORK_INPUT_PIXEL_NUM * 3;
const size_t CConfig::SAENETWORK_NEURO_NUM[] =
{
	1024, 
};
const size_t CConfig::SAENETWORK_LAYER_NUM = sizeof(SAENETWORK_NEURO_NUM) / sizeof(size_t);

const size_t CConfig::IMAGE_SIZE = CConfig::SAENETWORK_INPUT_NUM * sizeof(real);

//const real CConfig::STUDY_RATE = (real)0.00001; // pressure without NaN
const real CConfig::STUDY_RATE = (real)0.000001;
const real CConfig::STUDY_RATE_DECREASE_RATE = (real)1.0;

const size_t CConfig::TRAIN_SET_SIZE = 50;

const char CConfig::TRAIN_IMG_PATH[] = "E:\\Resources\\view_ex\\*.jpg";

const DWORD CConfig::LOAD_IMAGE_PERIOD = 300; // ms
