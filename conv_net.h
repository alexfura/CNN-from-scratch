#ifndef CONV_NET_H
#define CONV_NET_H

#include <vector>
#include <armadillo>
#include <QtDebug>
#include <cmath>
#include <limits>


using namespace arma;
using namespace std;


class ConvNet
{
public:
    ConvNet(uint n_features, uint n_outputs, uint kernel_size);
    void load(string path);
    void test_layers();
    void restore();
    // training
    void MBGD(uint epochs, uint batch_size, double learning_rate, double momentum);

    // testing
    uint predict(Mat<double>);
    void count_score();

    // saving model
    void save_model();

private:
    // layers
    Cube<double> maxpooling_layer(Cube <double> map);
    Cube<double> relu(Cube<double> map);
    Mat<double> softmax(Mat<double> layer);
    vec flatten(Cube<double> map);
    void fcLayer(vec flatten);
    Cube<double> ConvLayer(Mat<double> x, Cube<double> kernels);


    // hyperparameters
    uint kernel_size;
    uint n_features;
    uint n_output;

    // dimension converions
    Cube<double> to3d(vec flatten, uint rows, uint cols, uint slices);

    // initialization of weights
    void init_weigths();

    // layer outputs
    Cube<double> c1;
    Cube<double> a1;
    Cube<double> m1;
    vec f;
    Mat<double> h1;
    Mat<double> a2;
    Mat<double> h2;
    Mat<double> a3;

    // weights
    Cube<double> w1;
    Mat<double> w2;
    Mat<double> w3;

    // backprop and derivatives
    void backprop();
    Cube<double>MaxPoolingDerivative(Cube<double> pooledlayer, Cube<double> sigma);
    void setMax(Mat<double> &map, uint row, uint col, double max_value, double value);
    bool DoubleComp(double a, double b);
    Cube<double> relu_derivative(Cube<double> x);


    // dataset
    Cube<double> features;
    Mat<double> labels;
    Mat<double> encode_labels(Mat<double>);

    // need for calculus
    void feedforward(Mat<double>);
    void get_fc_gradients(Mat<double> y, Mat<double> o);
    void get_conv_gradient(Mat<double> x);
    Mat<double> softmax_der(Mat<double>);
    Cube<double> g1;
    Mat<double> g2, g3;
    Mat<double> s2, s3, s1;

    //destructor
    ~ConvNet();

};

#endif // CONV_NET_H
