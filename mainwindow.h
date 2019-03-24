#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <conv_net.h>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    ConvNet * net;

public Q_SLOTS:
    void cellSelected(int nRow, int nCol);
    void clear();

private:
    Ui::MainWindow *ui;
    void init_map();
    void predict();
    Mat<double> digit;
};

#endif // MAINWINDOW_H
