#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "conv_net.h"




MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);


    ConvNet * net = new ConvNet(784, 10, 5);

    net->test_layers();
    
}

MainWindow::~MainWindow()
{
    delete ui;

}
