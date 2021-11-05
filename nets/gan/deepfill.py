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

        # construct generator layers
        self.layer_1 = ConvolutionalLayer(filters, 7, stride=1)

        self.layer_2 = ConvolutionalLayer(2 * filters, 7, stride=2)
        self.layer_3 = ConvolutionalLayer(2 * filters, 7, stride=1)

        self.layer_4 = ConvolutionalLayer(4 * filters, 7, stride=2)
        self.layer_5 = ConvolutionalLayer(4 * filters, 7, stride=1)
        self.layer_6 = ConvolutionalLayer(4 * filters, 7, stride=2)
        self.layer_7 = ConvolutionalLayer(4 * filters, 7, stride=1)
        self.layer_8 = ConvolutionalLayer(4 * filters, 7, stride=1)
        self.layer_9 = ConvolutionalLayer(4 * filters, 7, dilation_rate=2)
        self.layer_10 = ConvolutionalLayer(4 * filters, 7, dilation_rate=4)
        self.layer_11 = ConvolutionalLayer(4 * filters, 7, dilation_rate=8)
        self.layer_12 = ConvolutionalLayer(4 * filters, 7, dilation_rate=16)

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = self.layer_9(x)
        x = self.layer_10(x)
        x = self.layer_11(x)
        x = self.layer_12(x)
        return x


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
