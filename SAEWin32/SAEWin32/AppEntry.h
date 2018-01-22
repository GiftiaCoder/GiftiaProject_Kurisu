#pragma once
#include "afxwin.h"
class CAppEntry :
	public CWinApp
{
public:
	CAppEntry();
	virtual ~CAppEntry();

public:
	virtual BOOL InitApplication() override;
	virtual BOOL InitInstance() override;
	virtual int Run() override;
	virtual int ExitInstance() override;
};

