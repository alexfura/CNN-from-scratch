#include "conv_net.h"


ConvNet::ConvNet(uint n_features, uint n_outputs, uint kernel_size)
{
    // n_feature - number of features
    // n_output - number of classes

    this->n_features = n_features;
    this->n_output = n_outputs;
    this->kernel_size = kernel_size;

//    this->init_weights();
}


void load(std::string path)
{

}

void ConvNet::to2d(Cube<double> &layer)
{
    // reshape 1d dataset to 2d
    // 1x784 to 1x28x28
}

Cube<double>ConvNet:: ConvLayer(Mat<double> x, Cube<double> kernels)
{
    Mat<double> sample;
    Cube<double> output;
    uint n_samples = static_cast<uint>(x.n_rows - kernels.n_rows) + 1;
    n_samples = 0;

    uint kernel_size = static_cast<uint>(kernels.n_rows);

    for (uint kernel = 0;kernel < kernels.n_slices;kernel++)
    {
        for(uint row = 0;row < n_samples;row++)
        {
            for (uint col = 0;col < n_samples;col++)
            {
                sample = x.submat(row, col, row + kernel_size- 1,
                                  col + kernel_size - 1);

                output.at(row, col, kernel) = accu(sample);
            }
        }
    }

    return output;
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


Cube<double> ConvNet::relu(Cube<double> map)
{
    map =  map.for_each([](mat::elem_type &val)
    {
            if(val < 0)
    {
            val = 0;
    }});

    return  map;
}


Row<double>ConvNet::flatten(Cube<double> map)
{
    Row<double> x_vector(map.n_cols * map.n_rows*map.n_slices);
    uint i =0;
    for (uint slice = 0;slice < map.n_slices;slice++)
    {
        for (uint row = 0;row < map.n_rows;row++)
        {
            for (uint col = 0;col < map.n_cols;col++)
            {
                x_vector.at(i) = map.at(row, col, slice);
                i++;
            }
        }
    }

    return  x_vector;
}

void ConvNet:: fcLayer(Row<double> flatten)
{

}


Mat<double>ConvNet:: softmax(Mat<double> layer)
{
    layer.for_each([](mat::elem_type &val){val = exp(val);});

    return  layer / layer.max();
}










