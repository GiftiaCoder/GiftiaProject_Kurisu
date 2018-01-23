#ifndef CONFIG_H
#define CONFIG_H

#include "libcu.h"

// show data config
#define GRID_SIZE (1.1F)
#define MAX_SAMPLE_EACH_COLUM (6)
#define SHOW_SAMPLE_NUM (MAX_SAMPLE_EACH_COLUM * 2)

extern const char SHOW_DATA_DIR[];

// neuro network config
#define STUDY_RATE (0.00001)
#define STUDY_RATE_DECREASE_RATE (0.98)

#define SAENETWORK_INPUT_NUM (96 * 96 * 4)
#define SAENETWORK_LAYER_NUM (1)
extern const size_t SAENETWORK_NEURO_NUM[];

#define IMAGE_SIZE (SAENETWORK_INPUT_NUM * sizeof(real))

#define TRAIN_SET_SIZE (64)

extern const char TRAIN_SET_PATH[];

#endif // CONFIG_H
