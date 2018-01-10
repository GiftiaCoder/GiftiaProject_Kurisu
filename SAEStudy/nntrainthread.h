#ifndef NNTRAINTHREAD_H
#define NNTRAINTHREAD_H

#include <QThread>
#include <QWidget>

#include "libcu.h"
#include "saenetwork.h"

#include "config.h"

class NNTrainThread : public QThread
{
private:
    class ImageLoadThread : public QThread
    {
    public:
        ImageLoadThread(NNTrainThread *parent);

    public:
        virtual void run() override;
    };

public:
    NNTrainThread(QWidget *pWidget, GLuint *pTexs, real **ppTexData, int texNum, const char *sampleDir);
    ~NNTrainThread();

public:
    virtual void run() override;

private:
    void loadRandomImage();

private:
    // display
    QWidget *m_pWidget;

    // network
    SAENetwork m_Network;

    // train set
    real **m_ppTrainSet;

    bool m_IsWaitSetUpdated;
    real **m_ppWaitSet;

    // show set
    real **m_ppTexsData;
    GLuint *m_pTexs;
    int m_TexNum;

    // image path
    char m_SampleDir[1024];

private:
    class SampleLoaderThread : public QThread
    {
    public:
        SampleLoaderThread(NNTrainThread *pParent);

    public:
       virtual void run() override;

    private:
        NNTrainThread *m_pParent;
    };
};

#endif // NNTRAINTHREAD_H
