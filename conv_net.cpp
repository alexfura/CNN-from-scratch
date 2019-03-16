#include "conv_net.h"

/*

INPUT -> CONV -> RELU -> MAXPOOLING -> FC

*/

ConvNet::ConvNet(uint n_features, uint n_outputs, uint kernel_size)
{
    // n_feature - number of features
    // n_output - number of classes
    // kernel size

    this->n_features = n_features;
    this->n_output = n_outputs;
    this->kernel_size = kernel_size;

//    this->init_weights();
}


void ConvNet::load(std::string path)
{
    mat raw = Mat<double>();
    raw.load(path);

    // expects, that y-label is first column

    // slice Raw matrix to inputs and labels
    // supposed that zero column in raw data is labels

    //    this->features = raw.submat(0, 1, raw.n_rows-1, raw.n_cols-1);
    this->labels = raw.submat(0, 0, raw.n_rows-1, 0);

}


void ConvNet:: init_weigths()
{
    arma_rng::set_seed_random();

    this->w1 = randu(this->kernel_size, this->kernel_size, 10);

    this->w2 = randu(this->kernel_size / 2 * this->kernel_size / 2, 100);

    this->w3 = randu(100, 10);
}


void ConvNet::to2d(Cube<double> &layer)
{
    // reshape 2d dataset to 3d
    // 1x784 to 1x28x28

}

Cube<double>ConvNet:: ConvLayer(Mat<double> x, Cube<double> kernels)
{
    Mat<double> sample;
    uint n_samples = static_cast<uint>(x.n_rows - kernels.n_rows) + 1;
    Cube<double> output(n_samples, n_samples, kernels.n_slices);

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


vec ConvNet::flatten(Cube<double> map)
{
    vec x_vector(map.n_cols * map.n_rows*map.n_slices);
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

void to3d(vec x, uint rows, uint cols, uint slices)
{
    Cube<double> output;
    if(rows * cols * slices != x.n_elem)
    {
        return;
    }
    //
    for (uint i = 0;i < x.n_elem;i++)
    {
        // reshape vec to 3d
        
    }
}

void ConvNet:: fcLayer(vec flatten)
{
    this->h1 = flatten * this->w2;
    // activate

    this->a3 = this->softmax(this->h1);

    this->h2 = this->w3 * this->a3;

    this->a4 = this->softmax(this->h2);
}


void ConvNet:: feedforward(Mat<double> x)
{
    this->c1 = this->ConvLayer(x, this->w1);

    this->a2 = this->relu(c1);
    
    this->m1 = this->maxpooling_layer(a2);
    
    this->f = this->flatten(m1);
    
    this->fcLayer(f);
}

Mat<double>ConvNet:: softmax_der(Mat<double> layer)
{
    return  this->softmax(layer) % (1 - this->softmax(layer));
}


void ConvNet:: get_fc_gradients(Mat<double> y, Mat<double> o)
{
    Mat<double> error = o - y;
    this->s3 = error % this->softmax_der(this->h2);
    this->g3 = s3 * this->a3;
    this->s2 = s3 * this->w3 * this->softmax_der(this->h1);
    this->g2 = s2 * this->f;
    this->s1 = s2 * this->w2;
    // g3 and g3
}


void ConvNet:: get_conv_gradient(Mat<double> x)
{
    // sigma = reshaped s1 (1x1440 -> 12x12-10)
    Cube<double> sigma;
    Cube<double> m1_der = this->MaxPoolingDerivative(sigma);
    this->g1 = this->ConvLayer(x, m1_der);
}

Mat<double> ConvNet:: softmax(Mat<double> layer)
{
    layer.for_each([](mat::elem_type &val){val = exp(val);});
    return  layer / layer.max();
}


Cube<double>ConvNet:: ConvDerivative(Mat<double> x, Cube<double> SigmaPrev)
{
    // 28x28 and 24x24xn -> 5x5xn
    return this->ConvLayer(x, SigmaPrev);
}

bool ConvNet::DoubleComp(double a, double b) {
    return fabs(a - b) < std::numeric_limits<double>::epsilon();
}

void ConvNet::setMax(Mat<double> &map, uint row, uint col, double max_value)
{
    for (uint i = row;i < row + 2;i++)
    {
        for(uint j = col;j  < col + 2;j++)
        {
            if(!this->DoubleComp(map.at(i, j), max_value))
            {
                map.at(i, j) = 0;
            }
            else{
                map.at(i, j) = 1;
            }
        }
    }
}

Cube<double>ConvNet:: MaxPoolingDerivative(Cube<double> PrevLayer)
{
    double max;
    for (uint slice = 0;slice < PrevLayer.n_slices;slice++)
    {
        for (uint row = 0;row < PrevLayer.n_rows;row+=2)
        {
            for (uint col = 0;col <  PrevLayer.n_cols;col+=2)
            {
                max = PrevLayer.slice(slice).submat(row, col, row+1, col+1).max();
                this->setMax(PrevLayer.slice(slice), row, col, max);
            }
        }
    }
    return PrevLayer;
}


void ConvNet::test_layers()
{
    arma_rng::set_seed_random();

    qDebug() <<"Testing max-pooling derivative";
    try {
        Cube<double> test = randu(8, 8, 1);
        test.print();
        Cube<double> detest = this->MaxPoolingDerivative(test);

        detest.print();
    } catch (const std::exception& e) {
        qDebug() <<e.what();
    }
}











