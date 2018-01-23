#include "ImageUtil.h"

#include "FreeImage.h"
#include "Config.h"

#pragma comment(lib, "FreeImage.lib")

bool CImageUtil::IsLoadable(const char *texpath)
{
	FREE_IMAGE_FORMAT fmt = FreeImage_GetFileType(texpath);
	FIBITMAP *pBitmap = FreeImage_Load(fmt, texpath);
	if (!pBitmap)
	{
		return false;
	}
	FreeImage_Unload(pBitmap);
	return true;
}

bool CImageUtil::LoadTexture(const char *texpath, real *dst)
{
	FREE_IMAGE_FORMAT fmt = FreeImage_GetFileType(texpath);
	FIBITMAP *pBitmap = FreeImage_Load(fmt, texpath);
	if (! pBitmap)
	{
		return false;
	}

	BYTE *pData = FreeImage_GetBits(pBitmap);
	for (int i = 0; i < CConfig::SAENETWORK_INPUT_NUM; i += 3)
	{
		dst[i + 0] = (real)0.8 * ((((real)(pData[i + 2])) / (real)128.0) - (real)1.0);
		dst[i + 1] = (real)0.8 * ((((real)(pData[i + 1])) / (real)128.0) - (real)1.0);
		dst[i + 2] = (real)0.8 * ((((real)(pData[i + 0])) / (real)128.0) - (real)1.0);
		//dst[i + 3] = (real)0.0;
	}

	FreeImage_Unload(pBitmap);

	return true;
}
