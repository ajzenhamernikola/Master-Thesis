#include "controllers/main_window.h"
#include "ui_main_window.h"

main_window::main_window(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::main_window)
{
    ui->setupUi(this);
}

main_window::~main_window()
{
    delete ui;
}

