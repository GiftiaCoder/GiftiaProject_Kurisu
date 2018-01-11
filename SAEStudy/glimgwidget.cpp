#include "glimgwidget.h"

#include <iostream>
#include <QDir>

#include <gl/GL.h>
#include <gl/GLU.h>

#include "imagehelper.h"
#include "nntrainthread.h"

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")
#pragma comment(lib, "user32.lib")

GLImgWidget::GLImgWidget(QWidget *parent, Qt::WindowFlags flags) :
    QOpenGLWidget(parent, flags)
{
    setWindowTitle("GLImgDisplayer");
}

GLImgWidget::~GLImgWidget()
{
    // TODO
}

void GLImgWidget::initializeGL()
{
    initializeOpenGLFunctions();

    glShadeModel(GL_SMOOTH);

    glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
    glClearDepth(1.0);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

    glEnable(GL_TEXTURE_2D);

    // textures
    std::cout << __FUNCSIG__ << "\t" << "prepare textures" << std::endl;
    glGenTextures(SHOW_SAMPLE_NUM, m_EncodeTexs);
    glGenTextures(SHOW_SAMPLE_NUM, m_DecodeTexs);

    // init textures data
    std::cout << __FUNCSIG__ << "\t" << "load texture data" << std::endl;
    QDir dir(SHOW_DATA_DIR);
    dir.setFilter(QDir::Files | QDir::NoSymLinks);
    QFileInfoList &list = dir.entryInfoList();

    real *pData = (real *)cuda_malloc_host(IMAGE_SIZE);
    for (int texIdx = 0; texIdx < SHOW_SAMPLE_NUM; ++texIdx)
    {
        createGLTexture(m_EncodeTexs[texIdx]);
        createGLTexture(m_DecodeTexs[texIdx]);

        ImageHelper::loadImage(list[texIdx].filePath().toLocal8Bit().data(), pData);

        m_pEncodeSet[texIdx] = (real *)cuda_malloc(IMAGE_SIZE);
        cuda_host_to_device(m_pEncodeSet[texIdx], pData, IMAGE_SIZE);
        gl_set_texture(m_EncodeTexs[texIdx], m_pEncodeSet[texIdx], IMAGE_SIZE);
    }
    cuda_free(pData);

    // begin training
    std::cout << __FUNCSIG__ << "\t" << "begin train" << std::endl;
    (new NNTrainThread(this, m_DecodeTexs, m_pEncodeSet, SHOW_SAMPLE_NUM, TRAIN_SET_PATH))->start();
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, 96, 96, 0, GL_BGRA_EXT, GL_FLOAT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
}
