#pragma once

#include "libcu.h"

class CConfig
{
public:
	const static size_t SAENETWORK_INPUT_PIXEL_NUM;
	const static size_t SAENETWORK_INPUT_NUM;
	const static size_t SAENETWORK_LAYER_NUM;
	const static size_t SAENETWORK_NEURO_NUM[];

public:
	const static size_t IMAGE_SIZE;

public:
	const static real STUDY_RATE;
	const static real STUDY_RATE_DECREASE_RATE;

public:
	const static size_t TRAIN_SET_SIZE;
	const static DWORD LOAD_IMAGE_PERIOD;

public:
	const static char TRAIN_IMG_PATH[];
};

