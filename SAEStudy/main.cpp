#include <QApplication>

#include "glimgwidget.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    (new GLImgWidget())->show();

    return a.exec();
}
