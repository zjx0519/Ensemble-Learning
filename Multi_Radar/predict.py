import random

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from Multi_Radar.PWFS_module import PWFS
from Multi_Radar.MassAtt_module import MassAtt
import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
random_seed = 250
random.seed(random_seed )  # set random seed for python
np.random.seed(random_seed )  # set random seed for numpy
tf.random.set_seed(random_seed )
custom_objects = {'PWFS': PWFS, 'MassAtt': MassAtt}
model = tf.keras.models.load_model(r'F:\chrome_download\LANMSFF-main\LANMSFF-main\Multi_Radar\model_best.h5', custom_objects=custom_objects)

X_test = []
Y_test = []
image_folder = r'F:\chrome_download\LANMSFF-main\LANMSFF-main\Datasets\Multi-Radar\Radar1-1\test'  # 替换成你的图像文件夹路径

# 获取所有文件
for label in range(4):
    label_folder = os.path.join(image_folder, str(label))
    for filename in os.listdir(label_folder):
        if filename.lower().endswith(".jpg"):  # 确保是 JPEG 文件
            image_path = os.path.join(label_folder, filename)
            image_gray = cv2.imread(image_path)
            if image_gray is not None:
                image_resized = cv2.resize(image_gray, (64, 64))
                X_test.append(image_resized)
                Y_test.append(label)

#
# Y_test = np.array(Y_test)
# X_test = np.array(X_test)
# # 将标签转换为整数
# label_values = np.unique(Y_test)
# label_to_int = {label: i for i, label in enumerate(label_values)}
# Y_test_int = np.array([label_to_int[label] for label in Y_test])
#
# # 将标签转换为 one-hot 编码
# num_classes = len(label_values)
# Y_test_onehot = to_categorical(Y_test_int, num_classes=num_classes)
# print("Y_test_int是：")
# print(Y_test_int)  # 应该输出 (64, ...)
# print("================================")
# print(X_test.shape)  # 应该输出 (64, ...)
# print("============================")
# # X_test = np.array(X_test,dtype=object)
# # X_test = np.array([cv2.resize(img, (64, 64)) for img in X_test])
# # Y_test = np.array(Y_test,dtype=object)
# # label_binarizer = LabelBinarizer()
# # Y_test_onehot = to_categorical(Y_test, num_classes=7)
# # 评估模型
# test_loss, test_acc = model.evaluate(X_test, Y_test_onehot, batch_size=32)
#
# print('Test Accuracy:', test_acc)
# print('Test Loss:', test_loss)
#
# # 进行预测
# predictions = model.predict(X_test)
#
# # 将预测结果转换为类别标签
# predicted_classes = np.argmax(predictions, axis=1)
#
# # 打印一些预测结果
# print("Predicted classes:", predicted_classes[:50])
#
# # 打印混淆矩阵
# cm = confusion_matrix(Y_test_int, predicted_classes)  # 使用整数标签
# print("Confusion Matrix:")
# print(cm)
#
# # 打印分类报告
# print("Classification Report:")
# print(classification_report(Y_test_int, predicted_classes))  # 使用整数标签
X_test=np.array(X_test)
Y_test=np.array(Y_test)
label_binarizer = LabelBinarizer()
Y_test_onehot = to_categorical(Y_test, num_classes=4)
print(Y_test_onehot.shape)

print("================================")
print(X_test.shape)  # 应该输出 (64, ...)
print(Y_test_onehot.shape)
print("============================")
test_loss, test_acc = model.evaluate(X_test, Y_test_onehot,batch_size=32)

print('Test Accuracy:', test_acc)
print('Test Loss:', test_loss)
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# 打印一些预测结果
print("Predicted classes:", predicted_classes[:50])
# for i, pred_class in enumerate(predicted_classes):
#     print(f"Image {i+1} predicted as expression {pred_class}")
cm = confusion_matrix(Y_test, predicted_classes)

# 打印混淆矩阵
print("Confusion Matrix:")
print(cm)

# 打印分类报告
print("Classification Report:")
print(classification_report(Y_test, predicted_classes))