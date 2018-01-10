#include "glimgwidget.h"

#include <gl/GL.h>
#include <gl/GLU.h>

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")
#pragma comment(lib, "user32.lib")

#include <iostream>

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
    // TODO
}
