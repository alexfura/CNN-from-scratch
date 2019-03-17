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
}

void ConvNet:: init_weigths()
{
    arma_rng::set_seed_random();

    this->w1 = randu(this->kernel_size, this->kernel_size, 10);

    this->w2 = randu(100, 1440);

    this->w3 = randu(10, 100);
}


void ConvNet::to2d(Cube<double> &layer)
{
    // reshape 2d dataset to 3d
    // 1x784 to 1x28x28
}

Cube<double>ConvNet:: to3d(vec flatten, uint rows, uint cols, uint slices)
{
    if(rows * cols * slices == flatten.n_rows)
    {
        qDebug() <<"Ok";
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

    this->a3 = this->softmax(this->h1);

    this->h2 = this->a3 * this->w3.t();

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
    this->g3 = s3.t() * this->a3;
    this->s2 = s3 * this->w3 % this->softmax_der(this->h1);
    this->g2 = s2.t() * this->f.t();
    this->s1 = s2 * this->w2;
    // g3 and g3
}


void ConvNet:: get_conv_gradient(Mat<double> x)
{
    // sigma = reshaped s1 (1x1440 -> 12x12-10)
    Cube<double> sigma = this->to3d(this->f, 12, 12, 10);
    qDebug() <<sigma.n_rows<<sigma.n_cols<<sigma.n_slices<<"sigma";

//    Cube<double> m1_der = this->MaxPoolingDerivative(sigma);
//    qDebug() <<m1.n_rows<<m1.n_cols<<m1.n_slices<<"M1";
//    this->g1 = this->ConvLayer(x, m1_der);
}

Mat<double> ConvNet:: softmax(Mat<double> layer)
{
    layer.for_each([](mat::elem_type &val){val = exp(val);});
    return  layer / accu(layer);
}


Cube<double>ConvNet:: ConvDerivative(Mat<double> x, Cube<double> SigmaPrev)
{
    // 28x28 and 24x24xn -> 5x5xn
    return this->ConvLayer(x, SigmaPrev);
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


void ConvNet::test_layers()
{
    arma_rng::set_seed_random();

    qDebug() <<"Testing max-pooling derivative";
    try {
        Cube<double> sigma = randu(4, 4, 10);
        // 24x24x10
        Cube<double> c1 = randu(8, 8, 10);
        Cube<double> der = this->MaxPoolingDerivative(c1, sigma);

        c1.slice(0).print(" C1 - SLICE");
        sigma.slice(0).print(" SIGMA ");
        der.slice(0).print("UPPOOLED MAP");
    } catch (const std::exception& e) {
        qDebug() <<e.what();
    }
//    qDebug() <<"Testing vec to 3d derivative";
//    // need for backprop
//    try {
//        vec test = randu(9);

//        test.print(" bla bla");

//        Cube<double> test_3d = this->to3d(test, 3, 3, 1);

//        test_3d.print("3d 5.0");

//    } catch (const std::exception& e) {
//        qDebug() <<e.what();
//    }

    // feeedforward
//    try {
//        vec x = randu(1440);
//        this->fcLayer(x);
//        this->a4.print();
//        qDebug() <<accu(this->a4);
//    } catch (const std::exception& e) {
//        qDebug() <<e.what();
//    }

//    // backward
//    try {
//        this->f = randu(1440);
//        this->fcLayer(this->f);
//        Mat<double> y = randu(1, 10);
//        Mat<double> o = randu(1, 10);

//        this->get_fc_gradients(y, o);
//        qDebug() <<this->g2.n_rows <<this->g2.n_cols <<"G2";
//        qDebug() <<this->g3.n_rows <<this->g3.n_cols <<"G3";
//    } catch (const std::exception& e) {
//        qDebug()<<e.what();
//    }
    // conv gradient
//    try {
//        this->f = randu(1440);
//        Mat<double> x = randu(28, 28);
//        this->get_conv_gradient(x);
//    } catch (const std::exception& e) {
//        qDebug() <<e.what();
//    }{}
}











