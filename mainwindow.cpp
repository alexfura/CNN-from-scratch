#include "mainwindow.h"
#include "ui_mainwindow.h"




MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    init_map();
    connect(ui->map, SIGNAL( cellEntered (int, int) ), this, SLOT( cellSelected( int, int ) ));
    connect(ui->clearButt, &QPushButton::clicked, this, [this]{this->clear();});
    net = new ConvNet(784, 10, 5);
    net->load("mnist_train.csv");
    net->MBGD(3, 12, 0.03, 0.92);
    net->load("mnist_test.csv");
    net->count_score();
    net->save_model();
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::init_map()
{
    ui->map->setRowCount(28);
    ui->map->setColumnCount(28);

    ui->map->setSelectionMode(QAbstractItemView::NoSelection);
    ui->map->verticalHeader()->hide();
    ui->map->horizontalHeader()->hide();

    ui->map->setShowGrid(false);
    ui->map->verticalHeader()->setMaximumSectionSize(10);
    ui->map->horizontalHeader()->setMaximumSectionSize(10);

    ui->map->verticalHeader()->setMinimumSectionSize(10);
    ui->map->horizontalHeader()->setMinimumSectionSize(10);



    for(int row = 0;row < ui->map->rowCount();row++)
    {
        for(int col = 0;col < ui->map->columnCount();col++)
        {
            ui->map->setItem(row, col, new QTableWidgetItem(""));
        }
    }
}

void MainWindow::cellSelected(int nRow, int nCol)
{
    this->ui->map->item(nRow, nCol)->setBackgroundColor(Qt::black);

    if(nRow != 27)
    {
        this->ui->map->item(nRow+1, nCol)->setBackgroundColor(Qt::black);
        this->digit.at(0, nCol + (nRow + 1) * 28) = 1;
    }
    if(nRow != 0)
    {
        this->ui->map->item(nRow - 1, nCol)->setBackgroundColor(Qt::black);
        this->digit.at(0, nCol + (nRow - 1) * 28) = 1;
    }
    if(nCol != 0)
    {
        this->ui->map->item(nRow, nCol - 1)->setBackgroundColor(Qt::black);
        this->digit.at(0, nCol - 1 + nRow * 28) = 1;
    }
    if(nCol != 27)
    {
        this->ui->map->item(nRow, nCol + 1)->setBackgroundColor(Qt::black);
        this->digit.at(0, nCol + 1 + nRow * 28) = 1;
    }
    this->digit.at(0, nCol + nRow * 28) = 1;
    this->predict();
}


void MainWindow::predict()
{
    qDebug()<<net->predict(digit)<<"Predicted";
}


void MainWindow::clear()
{
    for(int row = 0;row < ui->map->rowCount();row++)
    {
        for(int col = 0;col < ui->map->columnCount();col++)
        {
            this->ui->map->item(row, col)->setBackgroundColor(Qt::white);
        }
    }

    digit.zeros();
}
