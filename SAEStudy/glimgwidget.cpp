#include "glimgwidget.h"

#include <iostream>

#include <gl/GL.h>
#include <gl/GLU.h>

#include <QDir>

#include "imagehelper.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")
#pragma comment(lib, "user32.lib")

GLImgWidget::GLImgWidget(QWidget *parent, Qt::WindowFlags flags) :
    QOpenGLWidget(parent, flags),
    m_ppTrainSet(new real*[TRAIN_SET_SIZE]), m_ppWaitSet(new real*[TRAIN_SET_SIZE]), m_IsWaitSetUpdated(false),
    m_Network(SAENETWORK_INPUT_NUM, SAENETWORK_LAYER_NUM, SAENETWORK_NEURO_NUM)
{
    setWindowTitle("GLImgDisplayer");
}

GLImgWidget::~GLImgWidget()
{
    for (int i = 0; i < TRAIN_SET_SIZE; ++i)
    {
        cuda_free(m_ppTrainSet[i]);
        cuda_free(m_ppWaitSet[i]);
    }
    delete[] m_ppTrainSet;
    delete[] m_ppWaitSet;
    for (int i = 0; i < SHOW_SAMPLE_NUM; ++i)
    {
        cuda_free(m_pEncodeData[i]);
    }
}

void GLImgWidget::initializeGL()
{
    initializeOpenGLFunctions();

    set_cuda_device(0);

    glShadeModel(GL_SMOOTH);

    glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
    glClearDepth(1.0);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    // textures
    std::cout << __FUNCSIG__ << "\t" << "prepare textures" << std::endl;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(SHOW_SAMPLE_NUM, m_EncodeTexs);
    glGenTextures(SHOW_SAMPLE_NUM, m_DecodeTexs);
    for (int texIdx = 0; texIdx < SHOW_SAMPLE_NUM; ++texIdx)
    {
        createGLTexture(m_EncodeTexs[texIdx]);
        createGLTexture(m_DecodeTexs[texIdx]);
    }

    // begin training
    std::cout << __FUNCSIG__ << "\t" << "begin train" << std::endl;

    QDir dir(SHOW_DATA_DIR);
    dir.setFilter(QDir::Files | QDir::NoSymLinks);
    QFileInfoList &list = dir.entryInfoList();
    real *pData = (real *)cuda_malloc_host(IMAGE_SIZE);
    for (int texIdx = 0; texIdx < SHOW_SAMPLE_NUM; ++texIdx)
    {
        ImageHelper::loadImage(list[texIdx].filePath().toLocal8Bit().data(), pData);

        m_pEncodeData[texIdx] = (real *)cuda_malloc(IMAGE_SIZE);
        cuda_host_to_device(m_pEncodeData[texIdx], pData, IMAGE_SIZE);

        char errmsg[512] = { 0 };
        gl_set_texture(m_EncodeTexs[texIdx], m_pEncodeData[texIdx], IMAGE_SIZE / 2, errmsg);
        std::cout << __FUNCSIG__ << "\t" << errmsg << std::endl;
    }
    cuda_free((void *)pData);

    for (int i = 0; i < TRAIN_SET_SIZE; ++i)
    {
        m_ppTrainSet[i] = (real *)cuda_malloc(IMAGE_SIZE);
        m_ppWaitSet[i] = (real *)cuda_malloc(IMAGE_SIZE);
    }

    loadRandomImage();

    (new TrainThread(this))->start();
    (new SampleLoaderThread(this))->start();
}

void GLImgWidget::resizeGL(int w, int h)
{
    glViewport(0, 0, (GLint)w, (GLint)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(45.0, (GLfloat)w / (GLfloat)h, 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void GLImgWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glLoadIdentity();
    glTranslatef(-((GLfloat)6) * GRID_SIZE / 2.0F,
                 -((GLfloat)4) * GRID_SIZE / 2.0F, -8.0);

    int x = 0, y = 0;
    for (int texIdx = 0; texIdx < SHOW_SAMPLE_NUM; ++texIdx)
    {
        drawImage(x, y, m_EncodeTexs[texIdx]);
        drawImage(x, y + 1, m_DecodeTexs[texIdx]);

        if (++x == MAX_SAMPLE_EACH_COLUM)
        {
            x = 0;
            y += 2;
        }
    }
}

void GLImgWidget::networkTrainLoop()
{
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
        std::cout << __FUNCSIG__ << "\t" << "train" << std::endl;
        for (int trainCounter = 0; trainCounter < TRAIN_SET_SIZE; ++trainCounter)
        {
            m_Network.train(m_ppTrainSet[rand() % TRAIN_SET_SIZE], STUDY_RATE);
        }

        // display
        char errmsg[512] = { 0 };
        std::cout << __FUNCSIG__ << "\t" << "display" << std::endl;
        for (int idx = 0; idx < SHOW_SAMPLE_NUM; ++idx)
        {
            gl_set_texture(m_DecodeTexs[idx],
                           m_Network.decode(m_Network.encode(m_pEncodeData[idx])),
                           IMAGE_SIZE,
                           errmsg);
            std::cout << errmsg << std::endl;
        }

        update();
    }
}

void GLImgWidget::drawImage(int x, int y, GLuint texIdx)
{
    const static GLfloat VERTEXS[4][3] =
    {
        { 0.0f, 0.0f, 0.0f },
        { 1.0F, 0.0F, 0.0F },
        { 1.0F, 1.0F, 0.0F },
        { 0.0F, 1.0F, 0.0F },
    };

    glPushMatrix();

    glTranslatef(((GLfloat)x) * GRID_SIZE, ((GLfloat)y) * GRID_SIZE, 0);
    glBindTexture(GL_TEXTURE_2D, texIdx);
    glBegin(GL_QUADS);
        glTexCoord3fv(VERTEXS[0]);
        glVertex3fv(VERTEXS[0]);
        glTexCoord3fv(VERTEXS[1]);
        glVertex3fv(VERTEXS[1]);
        glTexCoord3fv(VERTEXS[2]);
        glVertex3fv(VERTEXS[2]);
        glTexCoord3fv(VERTEXS[3]);
        glVertex3fv(VERTEXS[3]);
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);

    glPopMatrix();
}

void GLImgWidget::createGLTexture(GLuint texIdx)
{
    glBindTexture(GL_TEXTURE_2D, texIdx);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, 96, 96, 0, GL_BGRA_EXT, GL_DOUBLE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    GL_RGBA16F_ARB;
}

void GLImgWidget::loadRandomImage()
{
    // images has just loaded
    if (m_IsWaitSetUpdated)
    {
        return;
    }

    // load random img to train wait set
    real *pHostBuff = (real *)cuda_malloc_host(IMAGE_SIZE);

    QDir dir(TRAIN_SET_PATH);
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

GLImgWidget::SampleLoaderThread::SampleLoaderThread(GLImgWidget *pParent) :
    m_pParent(pParent)
{
    // do nothing
}

void GLImgWidget::SampleLoaderThread::run()
{
    while (true)
    {
        QThread::sleep(5);
        m_pParent->loadRandomImage();
    }
}

GLImgWidget::TrainThread::TrainThread(GLImgWidget *pParent) :
    m_pParent(pParent)
{
    // do nothing
}

void GLImgWidget::TrainThread::run()
{
    m_pParent->networkTrainLoop();
}
