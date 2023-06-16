"""
official code:
https://github.com/google/automl/tree/master/efficientnetv2
"""

import itertools

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import AveragePooling2D, Lambda, Conv2D, Conv2DTranspose, Activation, Reshape, concatenate, Concatenate, BatchNormalization, ZeroPadding2D
from tensorflow.keras import backend as K

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}
def Upsample(tensor, size,name):
    '''bilinear upsampling'''
    #name = tensor.name.split('/')[0] + '_upsample'

    def bilinear_upsample(x, size):
        resized = tf.image.resize(
            images=x, size=size)
        return resized
    y = Lambda(lambda x: bilinear_upsample(x, size),
               output_shape=size, name=name)(tensor)
    return y
def Swish(inputs):
    x = tf.math.sigmoid(inputs) * inputs
    return x
class SE(layers.Layer):
    def __init__(self,
                 se_filters: int,
                 output_filters: int,
                 name: str = None):
        super(SE, self).__init__(name=name)
        self.swish=tf.nn.selu

        self.se_reduce = layers.Conv2D(filters=se_filters,
                                       kernel_size=1,
                                       strides=1,
                                       padding="same",
                                       use_bias=True,
                                       kernel_initializer=CONV_KERNEL_INITIALIZER,
                                       name="conv2d")

        self.se_expand = layers.Conv2D(filters=output_filters,
                                       kernel_size=1,
                                       strides=1,
                                       padding="same",
                                       activation="sigmoid",
                                       use_bias=True,
                                       kernel_initializer=CONV_KERNEL_INITIALIZER,
                                       name="conv2d_1")

    def call(self, inputs, **kwargs):
        # Tensor: [N, H, W, C] -> [N, 1, 1, C]
        se_tensor = tf.reduce_mean(inputs, [1, 2], keepdims=True)
        se_tensor = self.se_reduce(se_tensor)
        se_tensor = layers.Activation(self.swish)(se_tensor)
        se_tensor = self.se_expand(se_tensor)
        return se_tensor * inputs


class MBConv(layers.Layer):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float = 0.25,
                 drop_rate: float = 0.,
                 name: str = None):
        super(MBConv, self).__init__(name=name)

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)
        expanded_c = input_c * expand_ratio

        bid = itertools.count(0)
        get_norm_name = lambda: 'batch_normalization' + ('' if not next(
            bid) else '_' + str(next(bid) // 2))
        cid = itertools.count(0)
        get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
            next(cid) // 2))

        # 在EfficientNetV2中，MBConv中不存在expansion=1的情况所以conv_pw肯定存在
        assert expand_ratio != 1
        # Point-wise expansion
        self.expand_conv = layers.Conv2D(
            filters=expanded_c,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            name=get_conv_name())
        self.norm0 = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-3,
            name=get_norm_name())
        self.act0 = layers.Activation(tf.nn.selu)

        # Depth-wise convolution
        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=stride,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name="depthwise_conv2d")
        self.norm1 = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-3,
            name=get_norm_name())
        self.act1 = layers.Activation(tf.nn.selu)

        # SE
        num_reduced_filters = max(1, int(input_c * se_ratio))
        self.se = SE(num_reduced_filters, expanded_c, name="se")

        # Point-wise linear projection
        self.project_conv = layers.Conv2D(
            filters=out_c,
            kernel_size=1,
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name=get_conv_name())
        self.norm2 = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-3,
            name=get_norm_name())

        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            # Stochastic Depth
            self.drop_path = layers.Dropout(rate=drop_rate,
                                            noise_shape=(None, 1, 1, 1),  # binary dropout mask
                                            name="drop_path")

    def call(self, inputs, training=None):
        x = inputs

        x = self.expand_conv(x)
        x = self.norm0(x, training=training)
        x = self.act0(x)

        x = self.depthwise_conv(x)
        x = self.norm1(x, training=training)
        x = self.act1(x)

        x = self.se(x)

        x = self.project_conv(x)
        x = self.norm2(x, training=training)

        if self.has_shortcut:
            if self.drop_rate > 0:
                x = self.drop_path(x, training=training)

            x = tf.add(x, inputs)

        return x


class FusedMBConv(layers.Layer):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float = 0.,
                 name: str = None):
        super(FusedMBConv, self).__init__(name=name)
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        assert se_ratio == 0.

        self.has_shortcut = (stride == 1 and input_c == out_c)
        self.has_expansion = expand_ratio != 1
        expanded_c = input_c * expand_ratio

        bid = itertools.count(0)
        get_norm_name = lambda: 'batch_normalization' + ('' if not next(
            bid) else '_' + str(next(bid) // 2))
        cid = itertools.count(0)
        get_conv_name = lambda: 'conv2d' + ('' if not next(cid) else '_' + str(
            next(cid) // 2))

        if expand_ratio != 1:
            self.expand_conv = layers.Conv2D(
                filters=expanded_c,
                kernel_size=kernel_size,
                strides=stride,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                padding="same",
                use_bias=False,
                name=get_conv_name())
            self.norm0 = layers.BatchNormalization(
                axis=-1,
                momentum=0.9,
                epsilon=1e-3,
                name=get_norm_name())
            self.act0 = layers.Activation(tf.nn.selu)

        self.project_conv = layers.Conv2D(
            filters=out_c,
            kernel_size=1 if expand_ratio != 1 else kernel_size,
            strides=1 if expand_ratio != 1 else stride,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name=get_conv_name())
        self.norm1 = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-3,
            name=get_norm_name())

        if expand_ratio == 1:
            self.act1 = layers.Activation(tf.nn.selu)
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            # Stochastic Depth
            self.drop_path = layers.Dropout(rate=drop_rate,
                                            noise_shape=(None, 1, 1, 1),  # binary dropout mask
                                            name="drop_path")

    def call(self, inputs, training=None):
        x = inputs
        if self.has_expansion:
            x = self.expand_conv(x)
            x = self.norm0(x, training=training)
            x = self.act0(x)

        x = self.project_conv(x)
        x = self.norm1(x, training=training)
        if self.has_expansion is False:
            x = self.act1(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                x = self.drop_path(x, training=training)

            x = tf.add(x, inputs)

        return x


class Stem(layers.Layer):
    def __init__(self, filters: int=24, name: str = "Stem"):
        super(Stem, self).__init__(name=name)
        self.conv_stem = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=2,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name="conv2d")
        self.norm = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-3,
            name="batch_normalization")
        self.act = layers.Activation(tf.nn.selu)

    def call(self, inputs, training=None):
        
        x = self.conv_stem(inputs)
        x = self.norm(x, training=training)
        x = self.act(x)
        
        return x


class Head(layers.Layer):
    def __init__(self,
                 filters: int = 1280,
                 num_classes: int = 1000,
                 drop_rate: float = 0.,
                 name: str = None):
        super(Head, self).__init__(name=name)
        self.conv_head = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name="conv2d")
        self.norm = layers.BatchNormalization(
            axis=-1,
            momentum=0.9,
            epsilon=1e-3,
            name="batch_normalization")
        self.act = layers.Activation(tf.nn.selu)

        self.avg = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes,
                               kernel_initializer=DENSE_KERNEL_INITIALIZER)

        if drop_rate > 0:
            self.dropout = layers.Dropout(drop_rate)

    def call(self, inputs, training=None):
        x = self.conv_head(inputs)
        x = self.norm(x)
        x = self.act(x)
        x = self.avg(x)
 
        x = self.fc(x)
        return x
    

def EfficientNetV2(model_cnf: list,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 name: str = None,
                 img_height = 512,
                 img_width = 512):
   
        for cnf in model_cnf:
            assert len(cnf) == 8
        training=None
        stem_filter_num = model_cnf[0][4]
        stem = Stem(stem_filter_num)
        avgpool=AveragePooling2D
        conv=Conv2D
        bn=BatchNormalization
        acti=Activation
        upsample=Upsample
        concat=concatenate
        lamb=Lambda
        reshape=Reshape
        num_classes=num_classes

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []
        # Builds blocks.
        for cnf in model_cnf:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks.append(op(kernel_size=cnf[1],
                                      input_c=cnf[4] if i == 0 else cnf[5],
                                      out_c=cnf[5],
                                      expand_ratio=cnf[3],
                                      stride=cnf[2] if i == 0 else 1,
                                      se_ratio=cnf[-1],
                                      drop_rate=drop_connect_rate * block_id / total_blocks,
                                      name="blocks_{}".format(block_id)))
                block_id += 1
        img_input = layers.Input(shape=(img_width,img_height,3),dtype="float32")          
        x = Stem()(img_input)
        x_b=0
        i=0
        # call for blocks.``
        for _, block in enumerate(blocks):
            x = block(x, training=training)
            i=i+1
            if i==10:
                x_b=x
        #x = head(x, training=training)
        x_a = ASPP(x)
        x_a = upsample(tensor=x_a, size=[img_height // 4, img_width // 4],name=f'upsample2')
        
        x_b = conv(filters=48, kernel_size=1, padding='same',kernel_initializer='he_normal', name='low_level_projection', use_bias=False)(x_b)
        x_b = bn(name=f'bn_low_level_projection')(x_b)
        x_b = acti('relu', name='low_level_activation')(x_b)

        x = concat([x_a, x_b], name='decoder_concat')

        x = conv(filters=256, kernel_size=3, padding='same', activation='relu',
                kernel_initializer='he_normal', name='decoder_conv2d_1', use_bias=False)(x)
        x = bn(name=f'bn_decoder_1')(x)
        x = acti('relu', name='activation_decoder_1')(x)

        x = conv(filters=256, kernel_size=3, padding='same', activation='relu',kernel_initializer='he_normal', name='decoder_conv2d_2', use_bias=False)(x)
        x = bn(name=f'bn_decoder_2')(x)
        x = acti('relu', name='activation_decoder_2')(x)
        x = upsample(x, [img_height, img_width],name=f'upsample3')

        x = conv(num_classes, (1, 1), name='output_layer')(x)
        x = lamb(lambda xx:tf.image.resize(xx,[img_height,img_width]))(x)
        
        x = reshape((-1,num_classes))(x)
        x = acti('softmax')(x)
        model = Model(inputs=img_input, outputs=x, name='DeepLabV3_Plus')
        return model
def ASPP(tensor):
    dims = K.int_shape(tensor)

    y_pool = AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', name='pool_1x1conv2d', use_bias=False)(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('relu', name=f'relu_1')(y_pool)

    y_pool = Upsample(tensor=y_pool,size=[dims[1], dims[2]],name=f'upsample1')

    y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                kernel_initializer='he_normal', name='ASPP_conv2d_d1', use_bias=False)(tensor)
    y_1 = BatchNormalization(name=f'bn_2')(y_1)
    y_1 = Activation('relu', name=f'relu_2')(y_1)

    y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                kernel_initializer='he_normal', name='ASPP_conv2d_d6', use_bias=False)(tensor)
    y_6 = BatchNormalization(name=f'bn_3')(y_6)
    y_6 = Activation('relu', name=f'relu_3')(y_6)

    y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                kernel_initializer='he_normal', name='ASPP_conv2d_d12', use_bias=False)(tensor)
    y_12 = BatchNormalization(name=f'bn_4')(y_12)
    y_12 = Activation('relu', name=f'relu_4')(y_12)

    y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same',
                kernel_initializer='he_normal', name='ASPP_conv2d_d18', use_bias=False)(tensor)
    y_18 = BatchNormalization(name=f'bn_5')(y_18)
    y_18 = Activation('relu', name=f'relu_5')(y_18)

    y = concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
            kernel_initializer='he_normal', name='ASPP_conv2d_final', use_bias=False)(y)
    y = BatchNormalization(name=f'bn_final')(y)
    y = Activation('relu', name=f'relu_final')(y)
    
    return y    

    # def summary(self, input_shape=(224, 224, 3), **kwargs):
    #     x = Input(shape=input_shape)
    #     model = Model(inputs=[x], outputs=self.call(x, training=True))
    #     return model.summary()

    


def efficientnetv2_s(num_classes: int = 1000,img_height=512,img_width=512):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 300, eval_size: 384

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 1, 4, 48, 64, 0, 0],
                    [6, 3, 1, 4, 64, 128, 1, 0.25],
                    [9, 3, 2, 6, 128, 160, 1, 0.25]]
    

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.2,
                           name="efficientnetv2-s",
                           img_height=img_height,
                           img_width=img_width)
    return model


def efficientnetv2_m(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.3,
                           name="efficientnetv2-m")
    return model


def efficientnetv2_l(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4,
                           name="efficientnetv2-l")
    return model


# m = efficientnetv2_s()
# m.summary()
