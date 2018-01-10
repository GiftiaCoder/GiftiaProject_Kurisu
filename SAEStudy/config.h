#ifndef CONFIG_H
#define CONFIG_H

#include "libcu.h"

#define STUDY_RATE (0.00001)
#define STUDY_RATE_DECREASE_RATE (0.98)

#define SAENETWORK_INPUT_NUM (96 * 96 * 4)
#define SAENETWORK_LAYER_NUM (3)
const static size_t SAENETWORK_NEURO_NUM[] =
{
    32 * 32 * 2,
    16 * 16,
    4 * 4,
};

#define IMAGE_SIZE (SAENETWORK_INPUT_NUM * sizeof(real))

#define TRAIN_SET_SIZE (1024)

#endif // CONFIG_H
