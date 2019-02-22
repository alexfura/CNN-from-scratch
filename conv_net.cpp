#include "conv_net.h"


ConvNet::ConvNet(uint n_features, uint n_outputs, uint kernel_size)
{
    // n_feature - number of features
    // n_output - number of classes

    this->n_features = n_features;
    this->n_output = n_outputs;
    this->kernel_size = kernel_size;
}

void ConvNet::to2d(Cube<double> &layer)
{
    // reshape 1d dataset to 2d
}


Cube<double> ConvNet::conv_layer(Cube <double> features, vector<Cube<double>> kernels)
{
    Cube<double> map, subsample;
    uint n_samples = static_cast<uint>(features.slice(0).n_rows) - this->kernel_size + 1;
    Cube<double> output = zeros(n_samples, n_samples, kernels.size());
    for (uint kernel = 0;kernel < 1;kernel++)
    {
        for(uint row = 0;row < n_samples;row++)
        {
            for (uint col = 0;col < n_samples;col++)
            {
                subsample = features.subcube(row, col, 0, row + this->kernel_size - 1,
                                             col +this->kernel_size - 1, features.n_slices - 1);


                output.at(row, col, kernel) = accu(kernels.at(kernel) % subsample);
            }
        }
    }
    return  output;
}


Cube<double> ConvNet::maxpooling_layer(Cube <double> map)
{
    Cube<double> pooled_map = zeros(map.n_rows / 2, map.n_cols / 2, map.n_slices);
    uint m_row, m_col;
    Mat<double> sample;

    for (uint slice = 0;slice < pooled_map.n_slices;slice++)
    {
        m_row = 0;
        m_col = 0;
        for (uint row = 0;row < pooled_map.n_rows;row++)
        {
            for (uint col = 0;col <  pooled_map.n_cols;col++)
            {
                sample = map.slice(slice).submat(m_row, m_col, m_row+1, m_col+1);
                pooled_map.at(row, col, slice) = sample.max();
                if(m_col == map.n_cols - 2)
                {
                    m_row += 2;
                    m_col = 0;
                }
                else{
                    m_col +=2;
                }
            }
        }
    }

    return  pooled_map;
}


void ConvNet::relu(Cube<double> &map)
{
    map =  map.for_each([](mat::elem_type &val)
    {
    if(val < 0)
    {
      val = 0;
    }
    });
}


void ConvNet::test_layers()
{
    Cube<double> features = randu(28, 28, 10);
    vector<Cube<double>> kernels;

    arma_rng::set_seed_random();
    kernels.push_back(randu(5, 5, 1));



    vector<Cube<double>> kernels2;


    kernels2.push_back(randu(5, 5, kernels.size()));


    // testing conv

    Cube<double> l1 = this->conv_layer(features, kernels);

    l1.print();

    this->relu(l1);

    Cube<double> l2 = this->maxpooling_layer(l1);

    l2.print();

    Cube<double> l3 = this->conv_layer(l2, kernels2);

    l3.print();

    Cube<double> l4 = this->maxpooling_layer(l3);

    l4.print();

    qDebug() <<"Final: " <<l4.n_rows<<l4.n_cols<<l4.n_slices;
}






