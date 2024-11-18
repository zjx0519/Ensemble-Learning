import os
import shutil
import random

# 定义表情标签字典
emotion_labels = {
    0: "AN",
    1: "DI",
    2: "AF",
    3: "HA",
    4: "SA",
    5: "SU",
    6: "NE"
}
emotion_label_map = {v: k for k, v in emotion_labels.items()}

# 假设 KDEF 数据集当前所在的文件夹
kdef_dataset_folder = r'F:\chrome_download\LANMSFF-main\LANMSFF-main\Datasets\KDEF'
if not os.path.isdir(kdef_dataset_folder):
    raise ValueError(f"The directory {kdef_dataset_folder} does not exist.")

# 目标文件夹基础路径
base_output_folder = r'F:\chrome_download\LANMSFF-main\LANMSFF-main\Datasets\sorted_kdef_split'

# 创建目标文件夹（如果不存在）
for split in ['train', 'val', 'test']:
    for label in emotion_labels.values():
        os.makedirs(os.path.join(base_output_folder, split, label), exist_ok=True)

# 遍历 KDEF 数据集文件夹
for sub_folder_name in os.listdir(kdef_dataset_folder):
    sub_folder_path = os.path.join(kdef_dataset_folder, sub_folder_name)
    if os.path.isdir(sub_folder_path):
        file_list = [f for f in os.listdir(sub_folder_path) if f.lower().endswith('.jpg')]
        random.shuffle(file_list)  # 随机打乱文件列表
        # 计算每个数据集的文件数量
        total_files = len(file_list)
        train_size = int(0.8 * total_files)
        val_size = int(0.1 * total_files)  # 20%的一半，因为测试集也是 20%
        test_size = total_files - train_size - val_size  # 或者直接 int(0.2 * total_files)，结果一样

        # 分配文件到数据集
        train_files = file_list[:train_size]
        val_files = file_list[train_size:train_size + val_size]
        test_files = file_list[train_size + val_size:]

        # 移动文件到相应的数据集和表情标签文件夹
        for filename in train_files:
            label_str = filename[5:7]  # 根据实际情况调整索引以获取标签
            if label_str in emotion_label_map:
                emotion_label = emotion_label_map[label_str]
                if emotion_label in emotion_labels:
                    source_file_path = os.path.join(sub_folder_path, filename)
                    target_folder = os.path.join(base_output_folder, 'train', emotion_labels[emotion_label])
                    target_file_path = os.path.join(target_folder, filename)
                    if os.path.exists(source_file_path):
                        try:
                            shutil.move(source_file_path, target_file_path)
                        except Exception as e:
                            print(f"Error moving file {source_file_path}: {e}")

        for filename in val_files:
            label_str = filename[5:7]  # 根据实际情况调整索引以获取标签
            if label_str in emotion_label_map:
                emotion_label = emotion_label_map[label_str]
                if emotion_label in emotion_labels:
                    source_file_path = os.path.join(sub_folder_path, filename)
                    target_folder = os.path.join(base_output_folder, 'val', emotion_labels[emotion_label])
                    target_file_path = os.path.join(target_folder, filename)
                    if os.path.exists(source_file_path):
                        try:
                            shutil.move(source_file_path, target_file_path)
                        except Exception as e:
                            print(f"Error moving file {source_file_path}: {e}")

        for filename in test_files:
            label_str = filename[5:7]  # 根据实际情况调整索引以获取标签
            if label_str in emotion_label_map:
                emotion_label = emotion_label_map[label_str]
                if emotion_label in emotion_labels:
                    source_file_path = os.path.join(sub_folder_path, filename)
                    target_folder = os.path.join(base_output_folder, 'test', emotion_labels[emotion_label])
                    target_file_path = os.path.join(target_folder, filename)
                    if os.path.exists(source_file_path):
                        try:
                            shutil.move(source_file_path, target_file_path)
                        except Exception as e:
                            print(f"Error moving file {source_file_path}: {e}")
# 注意：打印语句可以放在循环外部，表示整个过程的完成
print("图像分类和数据集分割完成！")

# 注意：上面的代码中，val和test的for循环是空的，因为为了避免代码重复，
# 我没有在这里写出完整的移动逻辑。您应该为val和test文件集复制train文件集的移动逻辑，
# 只是将target_split分别更改为'val'和'test'。