#ifndef GLIMGWIDGET_H
#define GLIMGWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QThread>

#include "config.h"

class GLImgWidget : public QOpenGLWidget, public QOpenGLFunctions
{
    Q_OBJECT
public:
    GLImgWidget(QWidget *parent = Q_NULLPTR, Qt::WindowFlags flags = Qt::WindowFlags());
    ~GLImgWidget();

protected:
    virtual void initializeGL() override;
    virtual void resizeGL(int w, int h) override;
    virtual void paintGL() override;

private:
    void drawImage(int x, int y, GLuint texIdx);
    void createGLTexture(GLuint texIdx);

private:
    GLuint m_EncodeTexs[SHOW_SAMPLE_NUM];
    GLuint m_DecodeTexs[SHOW_SAMPLE_NUM];
    real *m_pEncodeSet[SHOW_SAMPLE_NUM];
};

#endif // GLIMGWIDGET_H
