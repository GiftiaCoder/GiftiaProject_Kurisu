#include "nntrainthread.h"

//#include <stdlib.h>

NNTrainThread::NNTrainThread(QWidget *pWidget, GLuint *pTexs, real **ppTexData, int texNum, const char *sampleDir) :
    m_pWidget(pWidget), m_pTexs(pTexs), m_TexNum(texNum), m_ppTexsData(ppTexData),
    m_Network(SAENETWORK_INPUT_NUM, SAENETWORK_LAYER_NUM, SAENETWORK_NEURO_NUM),
    m_ppTrainSet(new real*[TRAIN_SET_SIZE]), m_ppWaitSet(new real*[TRAIN_SET_SIZE]), m_IsWaitSetUpdated(false)
{
    loadRandomImage();
}

NNTrainThread::~NNTrainThread()
{
    // TODO delete all train set data
    delete[] m_ppTrainSet;
    delete[] m_ppWaitSet;
}

void NNTrainThread::run()
{
    while (true)
    {
        // choose set
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
    // TODO load random img to train wait set
}
