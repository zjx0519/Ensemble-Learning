#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, ReLU, Add,Dropout, Activation, Flatten, Input, PReLU,SeparableConv2D, Conv2DTranspose,concatenate,Convolution2D,ZeroPadding2D,Add,MaxPool2D
from tensorflow.keras.layers import Conv2D,Conv2DTranspose, Activation,MaxPooling3D, MaxPooling2D, BatchNormalization, UpSampling2D,AveragePooling2D,GlobalMaxPooling2D,GlobalAveragePooling2D
import random


def MassAtt(input_tensor, ratio=4):

        # Channel Attention Map
        num_input_channels = input_tensor.get_shape().as_list()[-1]
        # Squeeze operation: Global average pooling
        squeeze = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
        # Excitation operation: Two fully connected layers
        excitation = tf.keras.layers.Dense(units=num_input_channels // ratio, activation='relu')(squeeze)
        channel_att_map = tf.keras.layers.Dense(units=num_input_channels, activation='sigmoid')(excitation)


        # Spatial Attention Map
        spatial_attention = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_tensor)
        x= tf.keras.layers.Conv2D(filters= 2,kernel_size=3,kernel_initializer='he_uniform',activation='relu',strides=2,padding='same')(spatial_attention)
        x= tf.keras.layers.Conv2D(filters= 4,kernel_size=3,kernel_initializer='he_uniform',activation='relu',strides=2,padding='same')(x)
        x = Conv2DTranspose(4, (3, 3), activation='relu', padding='same',strides=2)(x)
        spatial_att_map = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same',strides=2)(x)

        # attention
        attention= channel_att_map * spatial_att_map * input_tensor

        return attention

