import keras.backend as K
import tensorflow as tf
from keras import initializers
from keras.engine import InputSpec
from keras.layers import Input, merge, ZeroPadding2D, Layer, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

MEAN, SCALE = 52.42393088150025, 0.022162079688563673


class Scale(Layer):
    '''Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:
        out = in * gamma + beta,
    where 'gamma' and 'beta' are the weights and biases larned.
    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''

    def __init__(self, weights=None, axis=-1, momentum=0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        # Compatibility with TensorFlow >= 1.0.0
        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))
        # self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        # self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def dense_net_121(img_rows, img_cols, color_type=1, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.5,
                  dropout_rate=0.0, weight_decay=0.0, dropout_fc=0.0, weights_path=None):
    '''
    DenseNet 121 Model for Keras
    Model Schema is based on 
    https://github.com/flyyufelix/DenseNet-Keras
    ImageNet Pretrained Weights 
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfMlRYb3YzV210VzQ
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfSTA4SHJVOHNuTXc
    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
        concat_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
        concat_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6, 12, 24, 16]  # For DenseNet-121

    # random crop
    X = img_input
    X = Lambda(tf.image.resize_image_with_crop_or_pad,
               output_shape=(224, 224, 3),
               arguments={'target_height': 224, 'target_width': 224})(X)

    # Normalize
    X = Lambda(lambda x: (x - MEAN) * SCALE, output_shape=(224, 224, 3))(X)

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(X)
    x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False, kernel_regularizer=l2(weight_decay))(
        x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn', beta_regularizer=l2(weight_decay),
                           gamma_regularizer=l2(weight_decay))(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate,
                             weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv' + str(final_stage) + '_blk_bn',
                           beta_regularizer=l2(weight_decay), gamma_regularizer=l2(weight_decay))(x)
    x = Scale(axis=concat_axis, name='conv' + str(final_stage) + '_blk_scale')(x)
    x = Activation('relu', name='relu' + str(final_stage) + '_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)
    x_newfc = Dropout(dropout_fc)(x_newfc)
    x_newfc = Dense(1, name='fc6', kernel_regularizer=l2(weight_decay))(x_newfc)
    x_newfc = Activation('sigmoid', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate
    model.compile(
        optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x1_bn',
                           beta_regularizer=l2(weight_decay), gamma_regularizer=l2(weight_decay))(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x1_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)
    x = Convolution2D(inter_channel, 1, 1, name=conv_name_base + '_x1', bias=False,
                      kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x2_bn',
                           beta_regularizer=l2(weight_decay), gamma_regularizer=l2(weight_decay))(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x2_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, 3, 3, name=conv_name_base + '_x2', bias=False, kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_bn',
                           beta_regularizer=l2(weight_decay), gamma_regularizer=l2(weight_decay))(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False,
                      kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4,
                grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis,
                            name='concat_' + str(stage) + '_' + str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter
