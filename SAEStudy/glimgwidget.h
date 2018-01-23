#ifndef GLIMGWIDGET_H
#define GLIMGWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QThread>

#include "config.h"
#include "saenetwork.h"

//#define GL_RGBA32F_ARB 0x8814

class GLImgWidget : public QOpenGLWidget, public QOpenGLFunctions
{
    Q_OBJECT
private:
    class SampleLoaderThread : public QThread
    {
    public:
        SampleLoaderThread(GLImgWidget *pParent);

    public:
       virtual void run() override;

    private:
        GLImgWidget *m_pParent;
    };

    class TrainThread : public QThread
    {
    public:
        TrainThread(GLImgWidget *pParent);

    public:
        virtual void run() override;

    private:
        GLImgWidget *m_pParent;
    };

public:
    GLImgWidget(QWidget *parent = Q_NULLPTR, Qt::WindowFlags flags = Qt::WindowFlags());
    ~GLImgWidget();

protected:
    virtual void initializeGL() override;
    virtual void resizeGL(int w, int h) override;
    virtual void paintGL() override;

public:
    void networkTrainLoop();

private:
    void drawImage(int x, int y, GLuint texIdx);
    void createGLTexture(GLuint texIdx);

    void loadRandomImage();

private:
    real *m_pEncodeData[SHOW_SAMPLE_NUM];
    GLuint m_EncodeTexs[SHOW_SAMPLE_NUM];
    GLuint m_DecodeTexs[SHOW_SAMPLE_NUM];

    // network
    SAENetwork m_Network;

    // train set
    real **m_ppTrainSet;
    real **m_ppWaitSet;
    bool m_IsWaitSetUpdated;
};

#endif // GLIMGWIDGET_H
