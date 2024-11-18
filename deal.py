import pandas as pd
import cv2
import numpy as np
import os

dataset_path = r'F:\chrome_download\challenges-in-representation-learning-facial-expression-recognition-challenge\train.csv'
image_size = (48, 48)

# 定义表情类别
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_faces():
    data = pd.read_csv(dataset_path)
    pixels = data['pixels'].tolist()
    emotions = data['emotion']

    for idx, (pixel_sequence, emotion) in enumerate(zip(pixels, emotions)):
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(48, 48)
        face = cv2.resize(face.astype('uint8'), image_size)

        # 获取对应的表情类别
        emotion_label = emotion_labels[emotion]

        # 确保对应的文件夹存在
        folder_path = os.path.join('fer2013_train', emotion_label)
        ensure_folder(folder_path)

        file_name = f"{emotion_label}_SIXU_{str(idx).zfill(6)}.jpg"
        file_path = os.path.join(folder_path, file_name)

        # 写入图像
        cv2.imwrite(file_path, face)


if __name__ == "__main__":
    save_faces()