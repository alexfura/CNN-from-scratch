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

    this->init_weigths();
}

ConvNet::~ConvNet()
{
}


void ConvNet::load(std::string path)
{
    mat raw = Mat<double>();
    raw.load(path);

    // expects, that y-label is first column

    // slice Raw matrix to inputs and labels
    // supposed that zero column in raw data is labels

    Mat<double> feature_1d = raw.submat(0, 1, raw.n_rows-1, raw.n_cols-1);
    Mat<double> buffer;
    this->features = zeros(28, 28, feature_1d.n_rows);
    for (uint i = 0;i < feature_1d.n_rows;i++)
    {
        buffer = feature_1d.row(i);
        buffer.reshape(28, 28);

        this->features.slice(i) = buffer.t();
        buffer.clear();
    }
    feature_1d.clear();
    this->labels = raw.submat(0, 0, raw.n_rows-1, 0);
    this->labels = this->encode_labels(this->labels);

    this->features /= 255;
//    this->features.for_each([](mat::elem_type& val){if(val > 0){val = 1;}});
}

Mat<double>ConvNet::encode_labels(Mat<double> labels)
{
    // n_rows x 1
    Mat<double> encoded = zeros(this->labels.n_rows, 10);

    for (uint i = 0;i < labels.n_rows;i++)
    {
        encoded.at(i, static_cast<uint>(labels.at(i, 0))) = 1;
    }

    return encoded;
}


void ConvNet:: init_weigths()
{
    arma_rng::set_seed_random();

    this->w1 = randu(this->kernel_size, this->kernel_size, 10);

    this->w2 = randu(100, 1440);

    this->w3 = randu(10, 100);
}

Cube<double>ConvNet:: to3d(vec flatten, uint rows, uint cols, uint slices)
{
    if(rows * cols * slices != flatten.n_rows)
    {
        throw std::logic_error("Conversion error!");
    }
    Cube<double> output = zeros(rows, cols, slices);
    Mat<double> buffer;
    uint step = rows * cols;
    for (uint i = 0;i < flatten.n_elem;i+= step)
    {
        buffer =  flatten.subvec(i, i + step - 1);
        buffer.reshape(rows, cols);

        output.slice(i/step) = buffer.t();
    }

    return output;
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

void ConvNet:: fcLayer(vec flatten)
{
    this->h1 = flatten.t() * this->w2.t();

    this->a2 = this->softmax(this->h1);

    this->h2 = this->a2 * this->w3.t();

    this->a3 = this->softmax(this->h2);
}


void ConvNet:: feedforward(Mat<double> x)
{
    this->c1 = this->ConvLayer(x, this->w1);

    this->a1 = this->relu(c1);
    
    this->m1 = this->maxpooling_layer(a1);
    
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
    this->g3 = this->s3.t() * this->a2;
    this->s2 = s3 * this->w3 % this->softmax_der(this->h1);
    this->g2 = s2.t() * this->f.t();
    this->s1 = s2 * this->w2;
}

Mat<double> ConvNet:: softmax(Mat<double> layer)
{
    double max = layer.max();
    layer.for_each([max](mat::elem_type &val){val = exp(val - max);});
    return  layer / accu(layer);
}

Cube<double>ConvNet::relu_derivative(Cube<double> x)
{
    // c1
    return  x.for_each([](mat::elem_type& val){
        if(val < 0)
        {
            val = 0;
        }
        else if(val > 0){
            val = 1;
        }
    });
}

void ConvNet:: get_conv_gradient(Mat<double> x)
{
    // x - feature map
    vec s1_vec = vectorise(s1);
    Cube<double> sigma = this->to3d(s1_vec, 12, 12, 10);
    Cube<double> uppool = this->MaxPoolingDerivative(this->a1, sigma);
    this->g1 = this->ConvLayer(x, uppool);
}

bool ConvNet::DoubleComp(double a, double b) {
    return fabs(a - b) < std::numeric_limits<double>::epsilon();
}

void ConvNet::setMax(Mat<double> &map, uint row, uint col, double max, double value)
{
    for (uint i = row;i < row + 2;i++)
    {
        for(uint j = col;j  < col + 2;j++)
        {
            if(!this->DoubleComp(map.at(i, j), max))
            {
                map.at(i, j) = 0;
            }
            else{
                map.at(i, j) = value;
            }
        }
    }
}

Cube<double>ConvNet::MaxPoolingDerivative(Cube<double> c1, Cube<double> sigma)
{
    // upsample matrix by inserting zeroes
    double max;
    double value = 0;
    for (uint slice = 0;slice < c1.n_slices;slice++)
    {
        for (uint row = 0;row < c1.n_rows;row+=2)
        {
            for (uint col = 0;col <  c1.n_cols;col+=2)
            {
                max = c1.slice(slice).submat(row, col, row+1, col+1).max();
                value = sigma.at(row/2, col/2, slice);
                this->setMax(c1.slice(slice), row, col, max, value);
            }
        }
    }
    return c1;
}


void ConvNet:: MBGD(uint epochs, uint batch_size, double learning_rate, double momentum)
{
    // mini-batch gradient descent
    Cube<double> g1_sum = zeros(this->w1.n_rows, this->w1.n_cols, this->w1.n_slices);
    Mat<double> g2_sum = zeros(this->w2.n_rows, this->w2.n_cols);
    Mat<double> g3_sum = zeros(this->w3.n_rows, this->w3.n_cols);

    Cube<double> v1;
    Cube<double> v2;
    Cube<double> v3;
    Cube<double> batch;
    double score = 0;
    for (uint epoch = 0;epoch < epochs;epoch++)
    {
        qDebug() <<"Epoch: "<<epoch;
        for (uint i = 0;i < this->features.n_slices;i+= batch_size)
        {
            batch = this->features.subcube(0, 0, i, this->features.n_rows - 1,
                                           this->features.n_cols - 1, i+ batch_size - 1);

            for (uint slice = 0;slice < batch.n_slices - 1;slice++)
            {
                this->feedforward(batch.slice(slice));
                if(this->labels.row(i+slice).index_max() == this->a3.index_max())
                {
                    score++;
                }
                this->get_fc_gradients(this->labels.row(i + slice), this->a3);
                this->get_conv_gradient(batch.slice(slice));
                g1_sum += this->g1;
                g2_sum += this->g2;
                g3_sum += this->g3;
            }
            // update weigths
            this->w1 -= learning_rate * g1_sum;
            this->w2 -= learning_rate * g2_sum;
            this->w3 -= learning_rate * g3_sum;

            g1_sum.zeros();
            g2_sum.zeros();
            g3_sum.zeros();
        }
        qDebug() << score / this->features.n_slices <<" % Total";
        score = 0;
    }
}

uint ConvNet::predict(Mat<double> input)
{
    this->feedforward(input);

    return this->a3.index_max();
}


void ConvNet:: count_score()
{
    double score = 0;
    for (uint i = 0;i < this->features.n_slices;i++)
    {
        if(predict(this->features.slice(i)) == this->labels.row(i).index_max())
        {
            score++;
        }
    }

    qDebug() <<score / this->features.n_slices <<" - % Total score";
}


void ConvNet::save_model()
{
    this->w1.save(hdf5_name("w1.h5"));
    this->w2.save(hdf5_name("w2.h5"));
    this->w3.save(hdf5_name("w3.h5"));
}


void ConvNet::restore()
{
    this->w1.load(hdf5_name("w1.h5"));
    this->w2.load(hdf5_name("w2.h5"));
    this->w3.load(hdf5_name("w3.h5"));
}


void ConvNet::test_layers()
{
    arma_rng::set_seed_random();
    try {
        this->feedforward(this->features.slice(0));
        this->get_fc_gradients(this->labels.row(0), this->a3);
        this->get_conv_gradient(this->features.slice(0));
        this->g1.print();
        this->g2.print();
        this->g3.print();
    } catch (const std::exception& e) {
        qDebug() <<e.what();
    }
}









