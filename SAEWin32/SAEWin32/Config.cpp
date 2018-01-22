#include "Config.h"

#include "libcu.h"

const size_t CConfig::SAENETWORK_INPUT_NUM = 96 * 96 * 4;
const size_t CConfig::SAENETWORK_NEURO_NUM[] =
{
	512, 
};
const size_t CConfig::SAENETWORK_LAYER_NUM = sizeof(SAENETWORK_NEURO_NUM) / sizeof(size_t);

const size_t CConfig::IMAGE_SIZE = CConfig::SAENETWORK_INPUT_NUM * sizeof(real);

const real CConfig::STUDY_RATE = (real)0.0001;
const real CConfig::STUDY_RATE_DECREASE_RATE = (real)0.99;

const size_t CConfig::TRAIN_SET_SIZE = 128;

const char CConfig::TRAIN_IMG_PATH[] = "E:\\Resources\\view_ex\\*.jpg";

const DWORD CConfig::LOAD_IMAGE_PERIOD = 5 * 1000; // ms
