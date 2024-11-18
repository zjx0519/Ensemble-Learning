import random

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from KDEF_NEW.PWFS_module import PWFS
from KDEF_NEW.MassAtt_module import MassAtt
import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
# 设置随机种子
random_seed = 200
np.random.seed(random_seed)

# 加载模型
custom_objects = {'PWFS': PWFS, 'MassAtt': MassAtt}
model = tf.keras.models.load_model(r'F:\chrome_download\LANMSFF-main\LANMSFF-main\KDEF_NEW\model_best.h5', custom_objects=custom_objects)
X_test = []
Y_test = []

image_folder = r'F:\chrome_download\LANMSFF-main\LANMSFF-main\Datasets\Radar4\multi-pos1\test\angle\box'

for label in range(8):
    label_folder = os.path.join(image_folder, str(label))
    for filename in os.listdir(label_folder):
        if filename.lower().endswith(".png"):  # 确保是 JPEG 文件
            image_path = os.path.join(label_folder, filename)
            image_gray = cv2.imread(image_path)
            if image_gray is not None:
                image_resized = cv2.resize(image_gray, (64, 64))
                X_test.append(image_resized)
                Y_test.append(label)


X_test=np.array(X_test)
Y_test=np.array(Y_test)
label_binarizer = LabelBinarizer()
Y_test_onehot = to_categorical(Y_test, num_classes=8)
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