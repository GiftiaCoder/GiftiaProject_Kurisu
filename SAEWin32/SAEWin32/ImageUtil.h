#pragma once

#include "libcu.h"

class CImageUtil
{
public:
	static bool IsLoadable(const char *texpath);
	static bool LoadTexture(const char *texpath, real *dst);
};

