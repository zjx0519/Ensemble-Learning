import os
import cv2
import numpy as np
import random
from random import randint
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

random_seed = 200
random.seed(random_seed)  # set random seed for python
np.random.seed(random_seed)  # set random seed for numpy
tf.random.set_seed(random_seed)
print("随机种子已设置")

X_train = []
Y_train = []
image_folder = r'F:\chrome_download\LANMSFF-main\LANMSFF-main\Datasets\Radar4\multi-pos1\train\angle\box'  # 替换成你的图像文件夹路径


def random_brightness(image):
    alpha = random.uniform(0.5, 1.5)
    return cv2.convertScaleAbs(image, alpha=alpha)


def random_contrast(image):
    alpha = random.uniform(0.5, 1.5)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


def add_gaussian_noise(image):
    mean = 0
    var = 10
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape).reshape(image.shape)
    gauss = gauss.astype('uint8')
    noisy = cv2.add(image, gauss)
    return noisy


for sub_folder_name in os.listdir(image_folder):
    sub_folder_path = os.path.join(image_folder, sub_folder_name)
    if os.path.isdir(sub_folder_path):
        for filename in os.listdir(sub_folder_path):
            if filename.endswith(".png"):
                image_path = os.path.join(sub_folder_path, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    image_resized = cv2.resize(image, (64, 64))  # 假设我们使用64x64作为最终尺寸
                    horizontal_flip = cv2.flip(image_resized, 1)
                    vertical_flip = cv2.flip(image_resized, 0)
                    z = random.randint(-15, 15)
                    rotation_matrix = cv2.getRotationMatrix2D((32, 32), z, 1.0)  # 中心点为32,32
                    rotated_image = cv2.warpAffine(image_resized, rotation_matrix, (64, 64))
                    brightness_image = random_brightness(image_resized)
                    contrast_image = random_contrast(image_resized)
                    noise_image = add_gaussian_noise(image_resized)

                    variants = [image_resized, horizontal_flip, vertical_flip, rotated_image, brightness_image,
                                contrast_image, noise_image]
                    for variant in variants:
                        X_train.append(variant)
                        label = sub_folder_name
                        print(label)
                        Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
label_binarizer = LabelBinarizer()
Y_train_onehot = label_binarizer.fit_transform(Y_train)

print("================================")
print(X_train.shape)  # 应该输出 (64, ...)
print(Y_train_onehot.shape)
print("============================")