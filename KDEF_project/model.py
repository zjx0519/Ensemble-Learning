#!/usr/bin/env python
# coding: utf-8

# In[1]:
import random
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, ReLU, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from KDEF_project.MassAtt_module import MassAtt
from KDEF_project.PWFS_module import PWFS
from KDEF_project.data_augmentation import X_train, x_val_fold, Y_train_onehot, y_val_fold_onehot
import absl.logging
absl.logging.set_verbosity(absl.logging.WARNING)  # 只显示警告和错误信息
random_seed = 42
random.seed(random_seed )  # set random seed for python
np.random.seed(random_seed )  # set random seed for numpy
tf.random.set_seed(random_seed )
print("随机种子已设置")

def two_path(input_tensor, filters, kernel_size, strides=(1, 1), padding='valid'):
    # Get input shape
    input_shape = input_tensor.get_shape().as_list()
    batch_size, height, width, input_channels = input_shape

    # Calculate the number of channels per group
    channels_per_group = input_channels // 2
    filters_per_group = filters // 2

    # Shuffle the channel indices randomly
    channel_indices = list(range(input_channels))
    random.shuffle(channel_indices)

    # Rearrange the input tensor based on shuffled channel indices
    input_tensor_shuffled = tf.gather(input_tensor, channel_indices, axis=-1)

    # Split input and filters into groups
    input_groups = tf.split(input_tensor_shuffled, 2, axis=-1)

    # H path- First stage of convolution
    convH1 = tf.keras.layers.Conv2D(filters=filters_per_group,
                                    kernel_size=kernel_size,
                                    kernel_initializer='he_uniform',
                                    strides=strides,
                                    padding=padding,
                                    kernel_regularizer=l2(0.001))(input_groups[0])
    convH1 = BatchNormalization()(convH1)
    convH1 = ReLU()(convH1)

    # L path- First stage of convolution
    convL1 = tf.keras.layers.Conv2D(filters=filters_per_group,
                                    kernel_size=kernel_size,
                                    kernel_initializer='he_uniform',
                                    dilation_rate=2,
                                    strides=strides,
                                    padding=padding,
                                    kernel_regularizer=l2(0.001))(input_groups[1])
    convL1 = BatchNormalization()(convL1)
    convL1 = ReLU()(convL1)

    # Concat first stage
    X1 = tf.concat([convH1, convL1], axis=-1)

    # H path- Second stage of convolution
    convH2 = tf.keras.layers.SeparableConv2D(filters=filters_per_group,
                                             kernel_size=kernel_size,
                                             kernel_initializer='he_uniform',
                                             strides=strides,
                                             padding=padding,
                                             kernel_regularizer=l2(0.001))(X1)
    convH2 = BatchNormalization()(convH2)
    convH2 = ReLU()(convH2)

    # L path- Second stage of convolution
    convL2 = tf.keras.layers.SeparableConv2D(filters=filters_per_group,
                                             kernel_size=kernel_size,
                                             kernel_initializer='he_uniform',
                                             dilation_rate=2,
                                             strides=strides,
                                             padding=padding,
                                             kernel_regularizer=l2(0.001))(X1)
    convL2 = BatchNormalization()(convL2)
    convL2 = ReLU()(convL2)

    # Concat second stage
    X2 = tf.concat([convH2, convL2], axis=-1)

    # H-path-Third stage of convolution
    convH3 = tf.keras.layers.Conv2D(filters=filters_per_group,
                                    kernel_size=kernel_size,
                                    kernel_initializer='he_uniform',
                                    strides=strides,
                                    padding=padding,
                                    kernel_regularizer=l2(0.001))(X2)
    convH3 = BatchNormalization()(convH3)
    convH3 = ReLU()(convH3)

    # L-path-Third stage of convolution
    convL3 = tf.keras.layers.Conv2D(filters=filters_per_group,
                                    kernel_size=kernel_size,
                                    kernel_initializer='he_uniform',
                                    dilation_rate=2,
                                    strides=strides,
                                    padding=padding,
                                    kernel_regularizer=l2(0.001))(X2)
    convL3 = BatchNormalization()(convL3)
    convL3 = ReLU()(convL3)

    # Final concat
    output_tensor = tf.concat([convH3, convL3], axis=-1)

    return output_tensor

# ********************************
# ********************************

input = tf.keras.Input(shape=(64, 64, 3))

# Block 1
b1 = tf.keras.layers.Conv2D(filters=66, kernel_size=(3, 3), kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(input)
b1 = BatchNormalization()(b1)
b1 = ReLU()(b1)
b1 = tf.keras.layers.SeparableConv2D(filters=66, kernel_size=(3, 3), kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(b1)
b1 = BatchNormalization()(b1)
b1 = ReLU()(b1)
b1 = tf.keras.layers.Conv2D(filters=66, kernel_size=(3, 3), kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(b1)
b1 = BatchNormalization()(b1)
b1 = MaxPooling2D(pool_size=2)(b1)
b1 = ReLU()(b1)
b1 = Dropout(0.4)(b1)

# Block 2
b2 = two_path(b1, filters=72, kernel_size=3, strides=(1, 1), padding='same')
b2 = MassAtt(b2, ratio=4)
b2 = Conv2D(72, kernel_size=(1, 1), kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(b2)
b2 = BatchNormalization()(b2)
b2 = MaxPooling2D(pool_size=2)(b2)
b2 = ReLU()(b2)
b2 = Dropout(0.4)(b2)

# Block 3
b3 = tf.keras.layers.Conv2D(filters=78, kernel_size=(3, 3), kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(b2)
b3 = BatchNormalization()(b3)
b3 = ReLU()(b3)
b3 = tf.keras.layers.SeparableConv2D(filters=78, kernel_size=(3, 3), kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(b3)
b3 = BatchNormalization()(b3)
b3 = ReLU()(b3)
b3 = tf.keras.layers.Conv2D(filters=78, kernel_size=(3, 3), kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(b3)
b3 = BatchNormalization()(b3)
b3 = MaxPooling2D(pool_size=2)(b3)
b3 = ReLU()(b3)
b3 = Dropout(0.4)(b3)

# Block 4
b4 = two_path(b3, filters=84, kernel_size=3, strides=(1, 1), padding='same')
b4 = MassAtt(b4, ratio=4)
b4 = Conv2D(84, kernel_size=(1, 1), kernel_initializer='he_uniform', padding='same', kernel_regularizer=l2(0.001))(b4)
b4 = BatchNormalization()(b4)
b4 = MaxPooling2D(pool_size=2)(b4)
b4 = ReLU()(b4)
b4 = Dropout(0.4)(b4)

b1 = PWFS()(b1)
b2 = PWFS()(b2)
b3 = PWFS()(b3)

b1 = GlobalAveragePooling2D()(b1)
b2 = GlobalAveragePooling2D()(b2)
b3 = GlobalAveragePooling2D()(b3)
b4 = GlobalAveragePooling2D()(b4)

f = tf.concat([b1, b2, b3, b4], axis=-1)

output = Dense(7, activation='softmax')(f)

model = tf.keras.Model(inputs=input, outputs=output)

model.summary()

# Compile the model
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])
#factor是学习率衰减因子，当9轮val_loss都没有变化，则lr*0.1
learning_rate_reducer = ReduceLROnPlateau('val_loss', factor=0.1, patience=9, verbose=1, mode='auto')
tensorboard = TensorBoard(log_dir='./logging')
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=14, verbose=1, mode='auto')
checkpoint = ModelCheckpoint('model_best.h5', save_best_only=True, monitor='val_loss', mode='min')
model.fit(X_train, Y_train_onehot,
          batch_size=16,
          epochs=300,
          verbose=1,
          validation_data=(x_val_fold, y_val_fold_onehot),
          callbacks=[learning_rate_reducer, tensorboard, early_stopper,checkpoint])
model.save_weights('model_weights.h5')
test_loss, test_acc = model.evaluate(np.array(x_val_fold), np.array(y_val_fold_onehot), batch_size=16)
print('test_pub_acc:', test_acc, 'test_loss', test_loss)
