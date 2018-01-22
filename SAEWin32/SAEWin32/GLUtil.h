#pragma once

#include "afxwin.h"

#include <gl\GL.h>
#include <gl\GLU.h>

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")

#define GL_RGBA32F_ARB (0x8814)
#define GL_RGB32F_ARB (0x8815)

class CGLUtil
{
public:
	static HGLRC CreateGLRC(HDC hDC);
	
private:
	static PIXELFORMATDESCRIPTOR s_stPixelFormatDescriptor;
};

