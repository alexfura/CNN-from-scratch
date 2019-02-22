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



private:

    Cube<double> conv_layer(Cube <double> features, vector<Cube<double>> kernels);
    Cube<double> maxpooling_layer(Cube <double> map);
    uint kernel_size;

    uint n_features;
    uint n_output;

    void  to2d(Cube<double> &layer);
    void relu(Cube<double> &map);


};

#endif // CONV_NET_H
