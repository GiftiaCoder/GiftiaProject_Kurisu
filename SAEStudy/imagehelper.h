#ifndef IMAGEHELPER_H
#define IMAGEHELPER_H

#include "libcu.h"

class ImageHelper
{
public:
    static void loadImage(const char *imgPath, real *pBuff);
};

#endif // IMAGEHELPER_H
