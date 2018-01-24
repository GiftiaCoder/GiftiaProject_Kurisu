#include "GLImgFrame.h"

#include "Util.h"
#include "resource.h"

#include <iostream>

CGLImgFrame::CGLImgFrame() :
	m_pTempTexData((float *)cuda_malloc(DISPLAY_PICTURE_DATA_LEN * sizeof(float))),
	//m_TrainImgPathList(0),
	m_IsWaitSetUpdated(false),
	m_ppTrainSet(new real*[CConfig::TRAIN_SET_SIZE]), m_ppWaitSet(new real*[CConfig::TRAIN_SET_SIZE]),
	m_Network(CConfig::SAENETWORK_INPUT_NUM, CConfig::SAENETWORK_LAYER_NUM, CConfig::SAENETWORK_NEURO_NUM),
	m_IsPause(false)
	//m_IsWndAlive(true)
{
	Create(NULL, "Lacia_SAE ver0.01");

	memset(m_Texs, 0, sizeof(m_Texs));
	memset(m_EncodeData, 0, sizeof(m_EncodeData));

	// train env
	for (int i = 0; i < CConfig::TRAIN_SET_SIZE; ++i)
	{
		m_ppTrainSet[i] = (real *)cuda_malloc(CConfig::IMAGE_SIZE);
		m_ppWaitSet[i] = (real *)cuda_malloc(CConfig::IMAGE_SIZE);
	}

	// gl env
	InitGLEnvironment();
	InitGLTextureData();

	/*CFileFind finder;
	BOOL hasNext = finder.FindFile(CConfig::TRAIN_IMG_PATH);
	while(hasNext)
	{
		hasNext = finder.FindNextFile();
		std::cout << finder.GetFilePath().GetString() << std::endl;
		if (CImageUtil::IsLoadable(finder.GetFilePath().GetString()))
		{
			std::cout << "load" << std::endl;
			m_TrainImgPathList.push_back(finder.GetFilePath().GetString());
		}
	}
	finder.Close();
	std::cout << m_TrainImgPathList.size() << std::endl;*/

	LoadRandTextures();
	CreateThread(NULL, 0, TrainNNProc, (LPVOID)this, 0, &m_TrainNNTid);
	CreateThread(NULL, 0, LoadImageProc, (LPVOID)this, 0, &m_LoadImgTid);
}

CGLImgFrame::~CGLImgFrame() 
{

	for (int i = 0; i < CConfig::TRAIN_SET_SIZE; ++i)
	{
		cuda_free(m_ppTrainSet[i]);
		cuda_free(m_ppWaitSet[i]);
	}
	delete[] m_ppTrainSet;
	delete[] m_ppWaitSet;

	glDeleteTextures(GRID_WIDTH * GRID_HEIGHT, (GLuint *)m_Texs);
	for (int i = 0; i < ENCODE_DATA_NUM; ++i)
	{
		if (m_EncodeData[i])
		{
			cuda_free(m_EncodeData[i]);
		}
	}
	cuda_free(m_pTempTexData);
}

BOOL CGLImgFrame::InitGLEnvironment()
{
	if ((m_hGLRC = CGLUtil::CreateGLRC(GetDC()->GetSafeHdc())) == INVALID_HANDLE_VALUE)
	{
		return FALSE;
	}
	if (! wglMakeCurrent(GetDC()->GetSafeHdc(), m_hGLRC))
	{
		return FALSE;
	}

	glShadeModel(GL_FLAT);
	glShadeModel(GL_SMOOTH);

	glClearColor(0.5F, 0.5F, 0.5F, 0.5F);
	glClearDepth(1.0);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);

	glEnable(GL_ALPHA_TEST);
	glEnable(GL_BLEND);
	glEnable(GL_TEXTURE_2D);

	return TRUE;
}

void CGLImgFrame::InitGLTextureData()
{
	// generate display texutres
	GLuint *pTextures = (GLuint *)m_Texs;
	glGenTextures(GRID_WIDTH * GRID_HEIGHT, pTextures);
	for (int i = 0; i < GRID_WIDTH * GRID_HEIGHT; ++i)
	{
		GLInitTexture(pTextures[i]);
	}

	// load encode data and set texture sources
	CFileFind finder;
	finder.FindFile(DISPLAY_PICTURE_DIR);
	
	real *pTempHostBuff = (real *)cuda_malloc_host(CConfig::IMAGE_SIZE);

	for (int x = 0, i = 0; x < GRID_WIDTH; x += 2)
	{
		for (int y = 0; y < GRID_HEIGHT; ++y)
		{
			m_EncodeData[i] = (real *)cuda_malloc(CConfig::IMAGE_SIZE);
			
			if (finder.FindNextFile())
			{
				CImageUtil::LoadTexture(finder.GetFilePath(), pTempHostBuff);
			}
			else
			{
				std::cout << "cannot find next file: " << i << std::endl;
			}
			cuda_memcpy(m_EncodeData[i], pTempHostBuff, CConfig::IMAGE_SIZE, host_to_device);

			translate_data_format(m_pTempTexData,
				m_EncodeData[i],
				CConfig::SAENETWORK_INPUT_PIXEL_NUM,
				real_to_float);

			gl_set_texture(m_Texs[x][y], m_pTempTexData, DISPLAY_PICTURE_DATA_LEN * sizeof(float));

			++i;
		}
	}

	cuda_free(pTempHostBuff);
}

DWORD WINAPI CGLImgFrame::LoadImageProc(LPVOID pPara)
{
	CGLImgFrame *pFrame = (CGLImgFrame *)pPara;
	pFrame->LoadImageProc();
	return 0;
}

DWORD WINAPI CGLImgFrame::TrainNNProc(LPVOID pPara)
{
	CGLImgFrame *pFrame = (CGLImgFrame *)pPara;
	pFrame->TrainNNProc();
	return 0;
}

void CGLImgFrame::LoadImageProc()
{
	while(GetSafeHwnd())
	//while (m_IsWndAlive)
	{
		if (! m_IsPause)
		{
			Sleep(CConfig::LOAD_IMAGE_PERIOD);
			LoadRandTextures();
			continue;
		}
		Sleep(3000);
	}
}

void CGLImgFrame::TrainNNProc()
{
	int l = 0;
	real studyRate = CConfig::STUDY_RATE;
	DWORD startTime = GetTickCount();
	while (true)
	//while (m_IsWndAlive)
	{
		if (!m_IsPause)
		{
			if (m_IsWaitSetUpdated)
			{
				real **ppTemp = m_ppTrainSet;
				m_ppTrainSet = m_ppWaitSet;
				m_ppWaitSet = ppTemp;

				m_IsWaitSetUpdated = false;

				printf("[TID:%d]traning set end swaping\n", GetCurrentThreadId());
			}

			//for (int i = 0; i < ENCODE_DATA_NUM; ++i)
			//{
			//	m_Network.Train(m_EncodeData[i], (real)(CConfig::STUDY_RATE * 0.01));
			//}

			for (int i = 0; i < CConfig::TRAIN_SET_SIZE; ++i)
			{
				m_Network.Train(m_ppTrainSet[i], studyRate);
			}
			//studyRate *= CConfig::STUDY_RATE_DECREASE_RATE;

			DWORD endTime = GetTickCount();
			printf("[TIME:%d]end training loop: %d\n", GetTickCount() - startTime, ++l);
			startTime = endTime;

			if (GetSafeHwnd())
			{
				SendMessage(WM_PAINT);
			}
			else
			{
				break;
			}

			continue;
		}
		Sleep(3000);
	}
}

void CGLImgFrame::LoadRandTextures()
{
	if (m_IsWaitSetUpdated)
	{
		return;
	}

	std::vector<std::string> pathList;
	LoadImageList(pathList);

	real *pTempHostBuff = (real *)cuda_malloc_host(CConfig::IMAGE_SIZE);
	
	for (int i = 0; i < CConfig::TRAIN_SET_SIZE; ++i)
	{
		CImageUtil::LoadTexture(pathList[rand() % pathList.size()].c_str(), pTempHostBuff);
		//CImageUtil::LoadTexture(m_TrainImgPathList[rand() % m_TrainImgPathList.size()].c_str(), pTempHostBuff);
		cuda_memcpy(m_ppWaitSet[i], pTempHostBuff, CConfig::IMAGE_SIZE, host_to_device);
	}

	cuda_free(pTempHostBuff);

	printf("[TID:%d]traning set end loading\n", GetCurrentThreadId());
	m_IsWaitSetUpdated = true;
}

void CGLImgFrame::SetDecodeTexData()
{
	//real *pTempHostData = (real *)cuda_malloc_host(CConfig::IMAGE_SIZE);
	for (int x = 1, i = 0; x < GRID_WIDTH; x += 2)
	{
		for (int y = 0; y < GRID_HEIGHT; ++y)
		{
			//cuda_memcpy(pTempHostData, m_Network.Decode(m_Network.Encode(m_EncodeData[i])), CConfig::IMAGE_SIZE, device_to_host);
			//for (int j = 0; j < 64; ++j)
			//{
			//	printf("%f ", pTempHostData[j]);
			//}
			//printf("\n");
			translate_data_format(m_pTempTexData, 
				m_Network.Decode(m_Network.Encode(m_EncodeData[i])),
				CConfig::SAENETWORK_INPUT_PIXEL_NUM, 
				real_to_float);
			gl_set_texture(m_Texs[x][y], m_pTempTexData, DISPLAY_PICTURE_DATA_LEN * sizeof(float));
			++i;
		}
	}
	//cuda_free(pTempHostData);
}

void CGLImgFrame::DoPaint()
{
	SetDecodeTexData();
	for (int x = 0; x < GRID_WIDTH; ++x)
	{
		for (int y = 0; y < GRID_HEIGHT; ++y)
		{
			PaintGLTexture(m_Texs[x][y], x, y);
		}
	}
}

int CGLImgFrame::OnCreate(LPCREATESTRUCT lpCs)
{
	int ret = CFrameWnd::OnCreate(lpCs);

	HICON hIcon = AfxGetApp()->LoadIcon(IDI_LACIA);
	SetIcon(hIcon, TRUE);
	SetIcon(hIcon, FALSE);

	return ret;
}

void CGLImgFrame::OnPaint()
{
	PAINTSTRUCT ps;
	BeginPaint(&ps);

	wglMakeCurrent(GetDC()->GetSafeHdc(), m_hGLRC);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glTranslatef(-((GLfloat)GRID_WIDTH) * GRID_SIZE / 2.0F,
		-((GLfloat)GRID_HEIGHT) * GRID_SIZE / 2.0F, -6.0F);

	DoPaint();

	SwapBuffers(GetDC()->GetSafeHdc());
	glFlush();

	EndPaint(&ps);
}

void CGLImgFrame::OnSize(UINT nType, int w, int h)
{
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	if (w <= h) {
		glOrtho(0.0f, 300.0f, 0.0f, 300.0f * h / w, 1.0f, -1.0f);
	}
	else {
		glOrtho(0.0f, 300.0f * w / h, 0.0f, 300.0f, 1.0f, -1.0f);
	}
	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 1.0, 1000.0);
	glMatrixMode(GL_MODELVIEW);
}

//void CGLImgFrame::OnClose()
//{
//	m_IsWndAlive = false;
//	CFrameWnd::OnClose();
//}

void CGLImgFrame::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	switch (nChar)
	{
	case VK_SPACE:
		TrainingPauseOrStart();
		break;
	default:
		break;
	}
}

BEGIN_MESSAGE_MAP(CGLImgFrame, CFrameWnd)
	ON_WM_CREATE()
	ON_WM_PAINT()
	ON_WM_SIZE()
	ON_WM_KEYDOWN()
	//ON_WM_CLOSE()
END_MESSAGE_MAP()

void CGLImgFrame::TrainingPauseOrStart()
{
	std::cout << (m_IsPause ? "training start" : "training pause") << std::endl;
	m_IsPause = (! m_IsPause);
}

void CGLImgFrame::GLInitTexture(GLuint tex)
{
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, 96, 96, 0, GL_BGRA_EXT, GL_FLOAT, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void CGLImgFrame::PaintGLTexture(GLuint tex, int x, int y)
{
	const static GLfloat vertexs[4][3] =
	{
		{ 0.0F, 0.0F, 0.0F, },
		{ 1.0F, 0.0F, 0.0F, },
		{ 1.0F, 1.0F, 0.0F, },
		{ 0.0F, 1.0F, 0.0F, },
	};

	glPushMatrix();

	glTranslatef(((GLfloat)x) * GRID_SIZE, ((GLfloat)y) * GRID_SIZE, 0);
	glBindTexture(GL_TEXTURE_2D, tex);
	glBegin(GL_QUADS);
	glTexCoord3fv(vertexs[0]);
	glVertex3fv(vertexs[0]);
	glTexCoord3fv(vertexs[1]);
	glVertex3fv(vertexs[1]);
	glTexCoord3fv(vertexs[2]);
	glVertex3fv(vertexs[2]);
	glTexCoord3fv(vertexs[3]);
	glVertex3fv(vertexs[3]);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);

	glPopMatrix();
}

void CGLImgFrame::LoadImageList(std::vector<std::string> &pathList)
{
	CFileFind fileFinder;
	BOOL hasNextFile = fileFinder.FindFile(CConfig::TRAIN_IMG_PATH);
	while (hasNextFile)
	{
		hasNextFile = fileFinder.FindNextFile();

		const char *path = fileFinder.GetFilePath().GetString();
		// if path is loadable, put it into the path list
		if (m_LoadableImageList.find(path) != m_LoadableImageList.end())
		{
			pathList.push_back(path);
		}
		// if path is unloadable
		else if (m_UnloadableImageList.find(path) != m_UnloadableImageList.end())
		{
			// do nothing
		}
		// if path is new, judge it
		else
		{
			if (CImageUtil::IsLoadable(path))
			{
				m_LoadableImageList.insert(path);
				pathList.push_back(path);
			}
			else
			{
				m_UnloadableImageList.insert(path);
			}
		}
	}
}
