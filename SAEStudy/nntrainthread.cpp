#include "nntrainthread.h"

#include <QDir>
#include <QFileInfo>

#include "imagehelper.h"

NNTrainThread::NNTrainThread(QWidget *pWidget, GLuint *pTexs, real **ppTexData, int texNum, const char *sampleDir) :
    m_pWidget(pWidget), m_pTexs(pTexs), m_TexNum(texNum), m_ppTexsData(ppTexData),
    m_Network(SAENETWORK_INPUT_NUM, SAENETWORK_LAYER_NUM, SAENETWORK_NEURO_NUM),
    m_ppTrainSet(new real*[TRAIN_SET_SIZE]), m_ppWaitSet(new real*[TRAIN_SET_SIZE]), m_IsWaitSetUpdated(false)
{
    for (int i = 0; i < TRAIN_SET_SIZE; ++i)
    {
        m_ppTrainSet[i] = cuda_malloc(IMAGE_SIZE);
        m_ppWaitSet[i] = cuda_malloc(IMAGE_SIZE);
    }

    strcpy_s(m_SampleDir, sampleDir);
    loadRandomImage();
}

NNTrainThread::~NNTrainThread()
{
    for (int i = 0; i < TRAIN_SET_SIZE; ++i)
    {
        cuda_free(m_ppTrainSet[i]);
        cuda_free(m_ppWaitSet[i]);
    }
    delete[] m_ppTrainSet;
    delete[] m_ppWaitSet;
}

void NNTrainThread::run()
{
    (new SampleLoaderThread(this))->start();
    while (true)
    {
        // exchange train set if wait set is updated
        if (m_IsWaitSetUpdated)
        {
            real **tempSet = m_ppTrainSet;
            m_ppTrainSet = m_ppWaitSet;
            m_ppWaitSet = tempSet;

            m_IsWaitSetUpdated = false;
        }

        // train
        for (int trainCounter = 0; trainCounter < TRAIN_SET_SIZE; ++trainCounter)
        {
            m_Network.train(m_ppTrainSet[rand() % TRAIN_SET_SIZE], STUDY_RATE);
        }

        // display
        for (int idx = 0; idx < m_TexNum; ++idx)
        {
            gl_set_texture(m_pTexs[idx],
                           m_Network.decode(m_Network.encode(m_ppTexsData[idx])),
                           IMAGE_SIZE);
        }
        m_pWidget->update();
    }
}

void NNTrainThread::loadRandomImage()
{
    // images has just loaded
    if (m_IsWaitSetUpdated)
    {
        return;
    }

    // load random img to train wait set
    real *pHostBuff = cuda_malloc_host(IMAGE_SIZE);

    QDir dir(m_SampleDir);
    dir.setFilter(QDir::Files | QDir::NoSymLinks);

    QFileInfoList &list = dir.entryInfoList();
    for (int i = 0; i < TRAIN_SET_SIZE; ++i)
    {
        QFileInfo &info = list[rand() % list.size()];
        ImageHelper::loadImage(info.filePath().toLocal8Bit().data(), pHostBuff);
        cuda_host_to_device(m_ppWaitSet[i], pHostBuff, IMAGE_SIZE);
    }
    m_IsWaitSetUpdated = true;

    cuda_free(pHostBuff);
}

NNTrainThread::SampleLoaderThread::SampleLoaderThread(NNTrainThread *pParent) :
    m_pParent(pParent)
{
    // do nothing
}

void NNTrainThread::SampleLoaderThread::run()
{
    while (true)
    {
        QThread::sleep(5);
        m_pParent->loadRandomImage();
    }
}
