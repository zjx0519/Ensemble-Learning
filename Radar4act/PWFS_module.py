
import tensorflow as tf
from keras.layers import Layer
import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, ReLU, Add,Dropout, Activation, Flatten, Input, PReLU,SeparableConv2D, Conv2DTranspose,concatenate,Convolution2D,ZeroPadding2D,Add,MaxPool2D
from tensorflow.keras.layers import Conv2D,Conv2DTranspose, Activation,MaxPooling3D, MaxPooling2D, BatchNormalization, UpSampling2D,AveragePooling2D,GlobalMaxPooling2D,GlobalAveragePooling2D
import random


class PWFS(Layer):
    def __init__(self,**kwargs):
        super(PWFS, self).__init__(**kwargs)

    def call(self, inputs):

        # Split feature map to 3 sub-group channel-wisely
        split1, split2, split3 = tf.split(inputs, num_or_size_splits=3, axis=-1)

        # Compute median using element-wise operations and minimum/maximum functions
        min_split = tf.minimum(tf.minimum(split1, split2), split3)
        max_split = tf.maximum(tf.maximum(split1, split2), split3)
        median_values = split1 + split2 + split3 - min_split - max_split

        # Averaging max and median sub-group
        average_values = 0.5 * (max_split + median_values)


        return average_values

    def get_config(self):
        # No additional hyperparameters to configure
        config = super(PWFS, self).get_config()
        return config

