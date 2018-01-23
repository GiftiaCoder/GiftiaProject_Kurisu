#-------------------------------------------------
#
# Project created by QtCreator 2018-01-01T14:38:47
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = SAEStudy
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
    glimgwidget.cpp \
    saenetwork.cpp \
    saelayer.cpp \
    imagehelper.cpp \
    config.cpp

HEADERS += \
    glimgwidget.h \
    libcu.h \
    saenetwork.h \
    saelayer.h \
    config.h \
    FreeImage.h \
    imagehelper.h

FORMS +=

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/./ -llibcu
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/./ -llibcud

INCLUDEPATH += $$PWD/.
DEPENDPATH += $$PWD/.

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/./liblibcu.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/./liblibcu.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/./libcu.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/./libcu.lib

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/./ -lFreeImage
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/./ -lFreeImaged

INCLUDEPATH += $$PWD/.
DEPENDPATH += $$PWD/.

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/./libFreeImage.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/./libFreeImaged.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/./FreeImage.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/./FreeImaged.lib
