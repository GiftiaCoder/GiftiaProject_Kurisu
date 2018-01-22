#include "GLUtil.h"

HGLRC CGLUtil::CreateGLRC(HDC hDC)
{
	int format = ChoosePixelFormat(hDC, &s_stPixelFormatDescriptor);
	if (format == 0)
	{
		return (HGLRC)INVALID_HANDLE_VALUE;
	}
	if (!SetPixelFormat(hDC, format, &s_stPixelFormatDescriptor))
	{
		return (HGLRC)INVALID_HANDLE_VALUE;
	}
	return wglCreateContext(hDC);
}

PIXELFORMATDESCRIPTOR CGLUtil::s_stPixelFormatDescriptor{
	sizeof(PIXELFORMATDESCRIPTOR),
	1,
	PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
	PFD_TYPE_RGBA,
	32,
	0, 0, 0, 0, 0, 0,
	0,
	0,
	0,
	0, 0, 0, 0,
	32,
	32,
	0,
	PFD_MAIN_PLANE,
	0,
	0, 0, 0,
};
