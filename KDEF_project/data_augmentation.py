
import os
import cv2
import numpy as np
import random
from random import randint
from tensorflow.keras.utils import to_categorical
# Assuming 'x_train' is a list of images with varying shapes
from sklearn.preprocessing import LabelBinarizer

X_train=[]
x_val_fold=[]
y_val_fold=[]
Y_train = []
image_folder = r'F:\chrome_download\LANMSFF-main\LANMSFF-main\Datasets\sorted_kdef_split\train'  # 替换成你的图像文件夹路径
val_folder=r"F:\chrome_download\LANMSFF-main\LANMSFF-main\Datasets\sorted_kdef_split\val"
for label in range(7):
    label_folder = os.path.join(image_folder, str(label))
    print(label)
    for filename in os.listdir(label_folder):
        if filename.lower().endswith(".jpg"):  # 确保是 JPEG 文件
            image_path = os.path.join(label_folder, filename)
            image = cv2.imread(image_path)
            if image is not None:
                # 处理图像变体
                image_resized = cv2.resize(image, (64, 64))  # 假设我们使用64x64作为最终尺寸
                horizontal_flip = cv2.flip(image_resized, 1)
                z = random.randint(-15, 15)
                rotation_matrix = cv2.getRotationMatrix2D((32, 32), z, 1.0)  # 中心点为32,32
                rotated_image = cv2.warpAffine(image_resized, rotation_matrix, (64, 64))

                # 注意：这里我们没有进行裁剪和再次缩放，以保持尺寸一致
                # 将变体添加到 X_train
                variants = [image_resized, horizontal_flip, rotated_image]  # 假设我们不使用裁剪的变体
                for variant in variants:
                    X_train.append(variant)
                    Y_train.append(label)

for label in range(7):
    label_folder = os.path.join(val_folder, str(label))
    for filename in os.listdir(label_folder):
        if filename.lower().endswith(".jpg"):  # 确保是 JPEG 文件
            image_path = os.path.join(label_folder, filename)
            image = cv2.imread(image_path)
            if image is not None:
                # 处理图像变体
                image_resized = cv2.resize(image, (64, 64))  # 假设我们使用64x64作为最终尺寸
                horizontal_flip = cv2.flip(image_resized, 1)
                z = random.randint(-15, 15)
                rotation_matrix = cv2.getRotationMatrix2D((32, 32), z, 1.0)  # 中心点为32,32
                rotated_image = cv2.warpAffine(image_resized, rotation_matrix, (64, 64))

                # 注意：这里我们没有进行裁剪和再次缩放，以保持尺寸一致
                # 将变体添加到 X_train
                variants = [image_resized, horizontal_flip, rotated_image]  # 假设我们不使用裁剪的变体
                for variant in variants:
                    x_val_fold.append(variant)
                    y_val_fold.append(label)

X_train=np.array(X_train)
Y_train = np.array(Y_train)
label_binarizer = LabelBinarizer()
# Y_train_onehot = label_binarizer.fit_transform(Y_train)
Y_train_onehot = to_categorical(Y_train, num_classes=7)
# 对验证集标签进行同样的处理
y_val_fold = np.array(y_val_fold)
y_val_fold_onehot = to_categorical(y_val_fold, num_classes=7)
x_val_fold=np.array(x_val_fold)

print("================================")
print(X_train.shape)  # 应该输出 (64, ...)
print(Y_train_onehot.shape)
print("============================")

print("================================")
print(x_val_fold.shape)  # 应该输出 (64, ...)
print(y_val_fold_onehot.shape)
print("============================")
# print(Y_train.dtype)

