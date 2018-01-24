#pragma once
#include "afxwin.h"

#include "libcu.h"
#include "SAENetwork.h"
#include "Config.h"

#include <unordered_set>
#include <vector>
#include <string>

#define GRID_SIZE (1.1F)
#define GRID_WIDTH (8)
#define GRID_HEIGHT (6)
#define ENCODE_DATA_NUM (GRID_WIDTH * GRID_HEIGHT / 2)
#define DISPLAY_PICTURE_DIR ("E:\\Resources\\view\\*.jpg")
#define DISPLAY_PICTURE_DATA_LEN (CConfig::SAENETWORK_INPUT_PIXEL_NUM * 4)

class CGLImgFrame :
	public CFrameWnd
{
public:
	CGLImgFrame();
	virtual ~CGLImgFrame();

private:
	BOOL InitGLEnvironment();
	void InitGLTextureData();

private:
	void SetDecodeTexData();
	void DoPaint();

	afx_msg void OnPaint();
	afx_msg void OnSize(UINT nType, int w, int h);
	//afx_msg void OnClose();
	afx_msg void OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags);

public:
	DECLARE_MESSAGE_MAP()

private:
	bool m_IsPause;
	void TrainingPauseOrStart();

private:
	void GLInitTexture(GLuint tex);
	void PaintGLTexture(GLuint tex, int x, int y);

private:
	bool m_IsWndAlive;
	static DWORD WINAPI LoadImageProc(LPVOID pPara);
	static DWORD WINAPI TrainNNProc(LPVOID pPara);
	DWORD m_LoadImgTid, m_TrainNNTid;

	void LoadImageProc();
	void TrainNNProc();

	bool m_IsWaitSetUpdated;
	real **m_ppTrainSet, **m_ppWaitSet;
	void LoadRandTextures();

private:
	CSAENetwork m_Network;

private:
	HGLRC m_hGLRC;

	GLuint m_Texs[GRID_WIDTH][GRID_HEIGHT];
	real *m_EncodeData[ENCODE_DATA_NUM];
	float *m_pTempTexData;

	//std::vector<std::string> m_TrainImgPathList;
private:
	void LoadImageList(std::vector<std::string> &pathList);
	std::unordered_set<std::string> m_LoadableImageList, m_UnloadableImageList;
};

