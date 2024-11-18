import random

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from KDEF_LAST.PWFS_module import PWFS
from KDEF_LAST.MassAtt_module import MassAtt
import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
random_seed = 102
random.seed(random_seed )  # set random seed for python
np.random.seed(random_seed )  # set random seed for numpy
tf.random.set_seed(random_seed )
print("随机种子已设置")
custom_objects = {'PWFS': PWFS, 'MassAtt': MassAtt}
model = tf.keras.models.load_model(r'F:\chrome_download\LANMSFF-main\LANMSFF-main\KDEF_LAST\model_best.h5', custom_objects=custom_objects)

X_test=[]
Y_test=[]
image_folder = r'F:\chrome_download\LANMSFF-main\LANMSFF-main\Datasets\Radar\test'  # 替换成你的图像文件夹路径
for sub_folder_name in os.listdir(image_folder):
    sub_folder_path = os.path.join(image_folder, sub_folder_name)
    if os.path.isdir(sub_folder_path):
        for filename in os.listdir(sub_folder_path):
            if filename.endswith(".jpg"):
                image_path = os.path.join(sub_folder_path, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    image_resized = cv2.resize(image, (64, 64))  # 假设我们使用64x64作为最终尺
                    X_test.append(image_resized)
                    label = sub_folder_name
                    Y_test.append(label)


X_test=np.array(X_test)
Y_test=np.array(Y_test)

label_binarizer = LabelBinarizer()
Y_test_onehot=label_binarizer.fit_transform(Y_test)
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
label_mapping = {'box': 0, 'down': 1, 'drink':2,'foot':3,'kick':4,'pick':5,'run':6,'sit':7,'up':8,'walk':9,'wave':10}
for label in Y_test:
    if label not in label_mapping:
        raise ValueError(f"Unknown label '{label}' found in test set.")
Y_test_int = np.array([label_mapping[label] for label in Y_test])
# 现在你可以计算混淆矩阵了
cm = confusion_matrix(Y_test_int, predicted_classes)


# 打印混淆矩阵
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(Y_test_int, predicted_classes, zero_division=1))