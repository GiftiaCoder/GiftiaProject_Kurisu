#ifndef GLIMGWIDGET_H
#define GLIMGWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QThread>

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
};

#endif // GLIMGWIDGET_H
