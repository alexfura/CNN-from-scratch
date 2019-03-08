#ifndef CONV_NET_H
#define CONV_NET_H

#include <vector>
#include <armadillo>
#include <QtDebug>


using namespace arma;
using namespace std;


class ConvNet
{
public:
    ConvNet(uint n_features, uint n_outputs, uint kernel_size);
    void load(string path);
    void test_layers();
    void feedforward(Cube<double> features);



private:
    // layers
    Cube<double> maxpooling_layer(Cube <double> map);
    Cube<double> relu(Cube<double> map);
    Mat<double> softmax(Mat<double> layer);
    Row <double> flatten(Cube<double> map);
    void fcLayer(Row<double> flatten);
    Cube<double> ConvLayer(Mat<double> x, Cube<double> kernels);


    // hyperparameters
    uint kernel_size;
    uint n_features;
    uint n_output;

    // dimension converions
    void  to2d(Cube<double> &layer);

    // initialization of weights
    void init_weights();

    // weights
    vector<Cube<double>> w1;

    // backprop and derivatives
    void backprop();
    Cube<double> ConvGrad(Cube<double> l, vector<Cube<double>> prev_gradient);
};

#endif // CONV_NET_H
