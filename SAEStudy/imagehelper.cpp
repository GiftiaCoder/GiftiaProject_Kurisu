#include "imagehelper.h"

#include "FreeImage.h"
#include "config.h"

void ImageHelper::loadImage(const char *imgPath, real *pBuff)
{
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(imgPath);
    FIBITMAP *pBitmap = FreeImage_Load(format, imgPath);

    BYTE *pData = FreeImage_GetBits(pBitmap);
    for (int i = 0, j = 0; i < SAENETWORK_INPUT_NUM; i += 4, j += 3)
    {
        pBuff[i + 0] = ((real)pData[j + 2]) / (real)256.0;
        pBuff[i + 1] = ((real)pData[j + 1]) / (real)256.0;
        pBuff[i + 2] = ((real)pData[j + 0]) / (real)256.0;
        pBuff[i + 3] = (real)1.0;
    }
    FreeImage_Unload(pBitmap);
}
