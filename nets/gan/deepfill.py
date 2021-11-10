import tensorflow as tf


class ConvolutionalLayer(tf.keras.layers.Layer):
    def __init__(self, filters, size, stride=1, dilation_rate=1, activation=tf.keras.activations.swish):
        self.filters = filters
        self.size = size
        self.stride = stride
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.conv_layer = tf.keras.layers.Conv2D(self.filters, self.size, self.stride, padding="same",
                                                 dilation_rate=self.dilation_rate, activation=self.activation)

    def call(self, incoming_tensor, training=False):
        result = self.conv_layer(incoming_tensor)
        # output layer?
        if self.filters == 3 or self.activation is None:
            return result

        result, y = tf.split(result, num_or_size_splits=2, axis=3)
        return self.activation(result) * tf.keras.activations.sigmoid(y)


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
        self.coarse_13 = ConvolutionalLayer(filters*2, 3, 1)
        self.coarse_14 = ConvolutionalLayer(filters//2, 3, 1)
        # deconv
        self.coarse_15 = ConvolutionalLayer(3, 3, 1, activation=None)

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

        # Final Network
        self.final_1 = ConvolutionalLayer(filters*4, 3, 1)
        self.final_2 = ConvolutionalLayer(filters*4, 3, 1)
        self.final_3 = ConvolutionalLayer(filters*2, 3, 1)
        self.final_4 = ConvolutionalLayer(filters//2, 3, 1)
        self.final_5 = ConvolutionalLayer(3, 3, 1, activation=None)
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
        # Coarse Network
        x_coarse = self.coarse_network(x)

        # Conv Branch
        x_conv = self.conv_branch(x_coarse)

        # Attention Branch
        x_att = self.att_branch(x_coarse)

        # Merge Results
        x_merge = tf.concat([x_conv, x_att], axis=3)

        # Final Network
        x_final = self.final_network(x_merge)

        return x_final

    def coarse_network(self, x):
        x = self.coarse_1(x)
        x = self.coarse_2(x)
        x = self.coarse_3(x)
        x = self.coarse_4(x)
        x = self.coarse_5(x)
        x = self.coarse_6(x)
        x = self.coarse_7(x)
        x = self.coarse_8(x)
        x = self.coarse_9(x)
        x = self.coarse_10(x)
        x = self.coarse_11(x)
        x = self.coarse_12(x)
        x = self.coarse_13(x)
        x = self.coarse_14(x)
        x = self.coarse_15(x)
        return x

    def conv_branch(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        x = self.conv_8(x)
        x = self.conv_9(x)
        x = self.conv_10(x)
        return x

    def att_branch(self, x):
        x = self.att_1(x)
        x = self.att_2(x)
        x = self.att_3(x)
        x = self.att_4(x)
        x = self.att_5(x)
        x = self.att_6(x)
        x = self.att_7(x)
        x = self.att_8(x)

    def final_network(self, x):
        x = self.final_1(x)
        x = self.final_2(x)
        x = self.final_3(x)
        x = self.final_4(x)
        x = self.final_5(x)

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


class Discriminator(tf.keras.Model):
    """
    Spectral-Normalized Markovian Discriminator (SN-PatchGAN)

    Determines whether the input image is real or fake
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        filter = 64
        kernel_size = 5
        stride = 2

        self.layer_1 = ConvolutionalLayer(filter, kernel_size, stride=stride)
        self.layer_2 = ConvolutionalLayer(2*filter, kernel_size, stride=stride)
        self.layer_3 = ConvolutionalLayer(4*filter, kernel_size, stride=stride)
        self.layer_4 = ConvolutionalLayer(4*filter, kernel_size, stride=stride)
        self.layer_5 = ConvolutionalLayer(4*filter, kernel_size, stride=stride)
        self.layer_6 = ConvolutionalLayer(4*filter, kernel_size, stride=stride)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        return x
