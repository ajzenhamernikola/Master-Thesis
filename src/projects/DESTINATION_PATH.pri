CONFIG(debug, release|debug) {
    CONFIG += BUILD_DEBUG
} else {
    CONFIG += BUILD_RELEASE
}

BUILD_DEBUG {
    build_path = debug
} else {
    build_path = release
}

DESTDIR = $$PWD/binaries/$$build_path
OBJECTS_DIR = $$PWD/build/$$build_path/.obj
MOC_DIR = $$PWD/build/$$build_path/.moc
RCC_DIR = $$PWD/build/$$build_path/.qrc
UI_DIR = $$PWD/build/$$build_path/.ui

DEFINES -= UNICODE
