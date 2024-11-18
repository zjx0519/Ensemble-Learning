import os

# 设置文件夹路径
folder_path =r'E:\桌面\Radar4act\test\test\zombie'
# 设置文件名前缀
prefix = 'Radar4_train_zombie_'

# 获取文件夹中所有jpg文件
files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 按文件名排序（可选）
files.sort()

# 批量重命名
for i, file in enumerate(files, start=1):
    new_name = f"{prefix}{i:04}.jpg"
    old_file = os.path.join(folder_path, file)
    new_file = os.path.join(folder_path, new_name)
    os.rename(old_file, new_file)
    print(f"Renamed {file} to {new_name}")