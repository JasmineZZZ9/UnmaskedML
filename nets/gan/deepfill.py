import tensorflow as tf
import numpy as np
from utils import *
from config import Config

FLAGS = Config('./inpaint.yml')
img_shape = FLAGS.img_shapes
IMG_HEIGHT = img_shape[0]
IMG_WIDTH = img_shape[1]

class ConvolutionalLayer(tf.keras.layers.Layer):
    def __init__(self, filters, size, stride=1, dilation_rate=1, activation=tf.keras.activations.swish):
        super(ConvolutionalLayer, self).__init__(name='')
        self.filters = filters
        self.size = size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.conv_layer = tf.keras.layers.Conv2D(self.filters, self.size, self.stride, padding="same",
                                                 dilation_rate=self.dilation_rate, activation=None)

    def call(self, incoming_tensor, training=False):
        result = self.conv_layer(incoming_tensor)
        # output layer?
        if self.filters == 3 or self.activation is None:
            return result

        result, y = tf.split(result, num_or_size_splits=2, axis=3)
        result = self.activation(result)
        return self.activation(result) * tf.keras.activations.sigmoid(y)


class gen_deconv_block(tf.keras.layers.Layer):
  def __init__(self, filters, multi=0):
    super(gen_deconv_block, self).__init__(name='')

    self.multi = multi
    # ΙΣΩΣ ΜΕ UPSAMPLING;
    #self.up2d = tf.keras.layers.UpSampling2D((2, 2), interpolation='nearest')
    self.ConvolutionalLayer = ConvolutionalLayer(filters, 3, 1)

  def call(self, input_tensor, training=False):
    #x = self.up2d(input_tensor)
    x = input_tensor
    if not self.multi:
        x = resize(x, func='nearest') #otan exo to generatormulticolumn den to thelo
    x = self.ConvolutionalLayer(x)
    return x

class Generator(tf.keras.Model):
    """
    First part of the GAN.

    Generates artificial faces
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        filters = 48

        # def __init__(self, filters, size, stride=1, dilation_rate=1, activation=tf.keras.activations.swish):

        # Coarse Network Layers
        self.coarse_1 = ConvolutionalLayer(filters, 5, 1)
        self.coarse_2 = ConvolutionalLayer(filters*2, 3, 2)
        self.coarse_3 = ConvolutionalLayer(filters*2, 3, 1)
        self.coarse_4 = ConvolutionalLayer(filters*4, 3, 2)
        self.coarse_5 = ConvolutionalLayer(filters*4, 3, 1)
        self.coarse_6 = ConvolutionalLayer(filters*4, 3, 1)
        self.coarse_7 = ConvolutionalLayer(filters*4, 3, dilation_rate=2)
        self.coarse_8 = ConvolutionalLayer(filters*4, 3, dilation_rate=4)
        self.coarse_9 = ConvolutionalLayer(filters*4, 3, dilation_rate=8)
        self.coarse_10 = ConvolutionalLayer(filters*4, 3, dilation_rate=16)
        self.coarse_11 = ConvolutionalLayer(filters*4, 3, 1)
        self.coarse_12 = ConvolutionalLayer(filters*4, 3, 1)
        # deconv
        # self.coarse_13 = ConvolutionalLayer(filters*2, 3, 1)
        # self.coarse_14 = ConvolutionalLayer(filters//2, 3, 1)
        self.coarse_13 = gen_deconv_block(filters*2)
        self.coarse_14 = ConvolutionalLayer(filters*2, 3, 1)


        # deconv
        # self.coarse_15 = ConvolutionalLayer(3, 3, 1, activation=None)
        self.coarse_15 = gen_deconv_block(filters)
        self.coarse_16 = ConvolutionalLayer(filters//2, 3, 1)
        self.coarse_17 = ConvolutionalLayer(3, 3, 1, activation=tf.keras.activations.tanh)

        # Conv Branch
        self.conv_1 = ConvolutionalLayer(filters, 5, 1)
        self.conv_2 = ConvolutionalLayer(filters, 3, 2)
        self.conv_3 = ConvolutionalLayer(filters*2, 3, 1)
        self.conv_4 = ConvolutionalLayer(filters*2, 3, 2)
        self.conv_5 = ConvolutionalLayer(filters*4, 3, 1)
        self.conv_6 = ConvolutionalLayer(filters*4, 3, 1)
        self.conv_7 = ConvolutionalLayer(filters*4, 3, dilation_rate=2)
        self.conv_8 = ConvolutionalLayer(filters*4, 3, dilation_rate=4)
        self.conv_9 = ConvolutionalLayer(filters*4, 3, dilation_rate=8)
        self.conv_10 = ConvolutionalLayer(filters*4, 3, dilation_rate=16)

        # Attention Branch
        self.att_1 = ConvolutionalLayer(filters, 5, 1)
        self.att_2 = ConvolutionalLayer(filters, 3, 2)
        self.att_3 = ConvolutionalLayer(filters*2, 3, 1)
        self.att_4 = ConvolutionalLayer(filters*4, 3, 2)
        self.att_5 = ConvolutionalLayer(filters*4, 3, 1)
        self.att_6 = ConvolutionalLayer(
            filters*4, 3, 1, activation=tf.keras.activations.relu)
        self.att_7 = ConvolutionalLayer(filters*4, 3, 1)
        self.att_8 = ConvolutionalLayer(filters*4, 3, 1)

        # combine layer?
        self.comb_1 = ConvolutionalLayer(filters*4, 3, 1)
        self.comb_2 = ConvolutionalLayer(filters*4, 3, 1)

        # Final Network
        #self.final_1 = ConvolutionalLayer(filters*4, 3, 1)
        #self.final_2 = ConvolutionalLayer(filters*4, 3, 1)
        #self.final_3 = ConvolutionalLayer(filters*2, 3, 1)
        #self.final_4 = ConvolutionalLayer(filters//2, 3, 1)
        #self.final_5 = ConvolutionalLayer(3, 3, 1, activation=None)

        self.final_1 = gen_deconv_block(filters * 2)
        self.final_2 = ConvolutionalLayer(filters * 2, 3, 1)
        self.final_3 = gen_deconv_block(filters)
        self.final_4 = ConvolutionalLayer(filters // 2, 3, 1)
        self.final_5 = ConvolutionalLayer(3, 3, 1, activation=tf.keras.activations.tanh)

        # self.layer_1 = ConvolutionalLayer(filters, 7, stride=1)

        # self.layer_2 = ConvolutionalLayer(2 * filters, 7, stride=2)
        # self.layer_3 = ConvolutionalLayer(2 * filters, 7, stride=1)

        # self.layer_4 = ConvolutionalLayer(4 * filters, 7, stride=2)
        # self.layer_5 = ConvolutionalLayer(4 * filters, 7, stride=1)
        # self.layer_6 = ConvolutionalLayer(4 * filters, 7, stride=2)
        # self.layer_7 = ConvolutionalLayer(4 * filters, 7, stride=1)
        # self.layer_8 = ConvolutionalLayer(4 * filters, 7, stride=1)
        # self.layer_9 = ConvolutionalLayer(4 * filters, 7, dilation_rate=2)
        # self.layer_10 = ConvolutionalLayer(4 * filters, 7, dilation_rate=4)
        # self.layer_11 = ConvolutionalLayer(4 * filters, 7, dilation_rate=8)
        # self.layer_12 = ConvolutionalLayer(4 * filters, 7, dilation_rate=16)

    def call(self, x, mask):
        """
        Generator for reconstructing masked areas

        Args:
            x: Image with masked faces [-1, 1] 
            mask: Image containing only masked region {0, 1}

        Returns:
            [-1, 1] Image with reconstructed faces in masked regions
        """
        # add
        xin = x
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x = tf.concat([x, ones_x, ones_x * mask], axis=3)

        # Coarse Network
        #x_coarse = self.coarse_network(x)
        x_coarse, mask_s = self.coarse_network(x, mask)

        # Conv Branch
        #x_conv = self.conv_branch(x_coarse)
        x_conv, xnow = self.conv_branch(x_coarse, mask, xin)

        # Attention Branch
        #x_att = self.att_branch(x_coarse)
        x_att, offset_flow = self.att_branch(x_conv, xnow, mask_s)

        # Merge Results
        x_merge = tf.concat([x_conv, x_att], axis=3)

        # Combine layers
        x_combine = self.combine_layer(x_merge)

        # Final Network
        x_final = self.final_network(x_combine)

        return x_coarse, x_final, offset_flow

    def coarse_network(self, x, mask):
        x = self.coarse_1(x)
        x = self.coarse_2(x)
        x = self.coarse_3(x)
        x = self.coarse_4(x)
        x = self.coarse_5(x)
        x = self.coarse_6(x)
        # add
        mask_s = resize_mask_like(mask, x)
        x = self.coarse_7(x)
        x = self.coarse_8(x)
        x = self.coarse_9(x)
        x = self.coarse_10(x)
        x = self.coarse_11(x)
        x = self.coarse_12(x)
        x = self.coarse_13(x)
        x = self.coarse_14(x)
        x = self.coarse_15(x)
        x = self.coarse_16(x)
        x = self.coarse_17(x)
        return x, mask_s

    def conv_branch(self, x, mask, xin):
        # add
        x = x * mask + xin[:, :, :, 0:3] * (1. - mask)
        x.set_shape(xin[:, :, :, 0:3].get_shape().as_list())
        xnow = x

        x = self.conv_1(xnow)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        x = self.conv_8(x)
        x = self.conv_9(x)
        x = self.conv_10(x)
        return x, xnow

    def att_branch(self, x, xnow, mask_s):
        x = self.att_1(xnow)
        x = self.att_2(x)
        x = self.att_3(x)
        x = self.att_4(x)
        x = self.att_5(x)
        x = self.att_6(x)

        x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)

        x = self.att_7(x)
        x = self.att_8(x)
        return x, offset_flow

    def combine_layer(self, x):
        x = self.comb_1(x)
        x = self.comb_2(x)
        return x

    def final_network(self, x):
        x = self.final_1(x)
        x = self.final_2(x)
        x = self.final_3(x)
        x = self.final_4(x)
        x = self.final_5(x)
        return x

    # def call(self, inputs):
    #     x = self.layer_1(inputs)
    #     x = self.layer_2(x)
    #     x = self.layer_3(x)
    #     x = self.layer_4(x)
    #     x = self.layer_5(x)
    #     x = self.layer_6(x)
    #     x = self.layer_7(x)
    #     x = self.layer_8(x)
    #     x = self.layer_9(x)
    #     x = self.layer_10(x)
    #     x = self.layer_11(x)
    #     x = self.layer_12(x)
    #     return x

class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_v',
                                 dtype=tf.float32)

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_u',
                                 dtype=tf.float32)

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = self.v  # init v vector

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)

        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)


class discriminator_SN(tf.keras.layers.Layer):
  def __init__(self, filters, size=5, stride=2):
    super(discriminator_SN, self).__init__(name='')
    self.sn_conv = SpectralNormalization(
        tf.keras.layers.Conv2D(filters, size, strides=stride, padding='same',
                               activation=tf.keras.layers.LeakyReLU(alpha=0.2))
    )

  def call(self, inputs):
    return self.sn_conv(inputs)

class Discriminator(tf.keras.Model):
    """
    Spectral-Normalized Markovian Discriminator (SN-PatchGAN)

    Determines whether the input image is real or fake
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        filter = 64
        #kernel_size = 5
        #stride = 2

        self.sn_conv_1 = discriminator_SN(filter)
        self.sn_conv_2 = discriminator_SN(2*filter)
        self.sn_conv_3 = discriminator_SN(4*filter)
        self.sn_conv_4 = discriminator_SN(4*filter)
        self.sn_conv_5 = discriminator_SN(4*filter)
        self.sn_conv_6 = discriminator_SN(4*filter)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        inputs = self.sn_conv_1(inputs)
        inputs = self.sn_conv_2(inputs)
        inputs = self.sn_conv_3(inputs)
        inputs = self.sn_conv_4(inputs)
        inputs = self.sn_conv_5(inputs)
        inputs = self.sn_conv_6(inputs)
        return self.flatten(inputs)
