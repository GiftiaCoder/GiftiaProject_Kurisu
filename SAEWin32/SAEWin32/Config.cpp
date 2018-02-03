#include "Config.h"

#include "libcu.h"

const size_t CConfig::INPUT_IMAGE_WIDTH = 64;
const size_t CConfig::INPUT_IMAGE_HEIGHT = 64;

const size_t CConfig::SAENETWORK_INPUT_PIXEL_NUM = INPUT_IMAGE_WIDTH * INPUT_IMAGE_HEIGHT;
const size_t CConfig::SAENETWORK_INPUT_NUM = SAENETWORK_INPUT_PIXEL_NUM * 3;
const size_t CConfig::SAENETWORK_NEURO_NUM[] =
{
	1024 * 8, 
};
const size_t CConfig::SAENETWORK_LAYER_NUM = sizeof(SAENETWORK_NEURO_NUM) / sizeof(size_t);

const size_t CConfig::IMAGE_SIZE = CConfig::SAENETWORK_INPUT_NUM * sizeof(real);

//const real CConfig::STUDY_RATE = (real)0.00001; // pressure without NaN
const real CConfig::STUDY_RATE = (real)0.0000984;
const real CConfig::STUDY_RATE_DECREASE_RATE = (real)0.1;

const size_t CConfig::TRAIN_SET_SIZE = 100;

const char CConfig::TRAIN_IMG_PATH[] = "E:\\Resources\\view_ex\\*.jpg";

const DWORD CConfig::LOAD_IMAGE_PERIOD = 300; // ms
