import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
def load_dataset(data_dir):
    image_paths = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            image_paths.append(image_path)
            labels.append(int(label))
    return image_paths, labels

train_dir = r'F:\chrome_download\LANMSFF-main\LANMSFF-main\Datasets\Multi-Radar\Radar1-1\train'
test_dir = r'F:\chrome_download\LANMSFF-main\LANMSFF-main\Datasets\Multi-Radar\Radar1-1\test'

train_image_paths, train_labels = load_dataset(train_dir)
test_image_paths, test_labels = load_dataset(test_dir)

train_dataset = CustomDataset(train_image_paths, train_labels, transform)
test_dataset = CustomDataset(test_image_paths, test_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 定义模型
from JIcheng_Learning.ShuffleNetV2 import shufflenet_v2_x1_0
from JIcheng_Learning.MobileNetV2 import MobileNetV2
from JIcheng_Learning.MobileNetV3 import mobilenet_v3_large

def create_model(model_type, num_classes):
    if model_type == 'shufflenet':
        model = shufflenet_v2_x1_0(num_classes=num_classes)
    elif model_type == 'mobilenetv2':
        model = MobileNetV2(num_classes=num_classes)
    elif model_type == 'mobilenetv3':
        model = mobilenet_v3_large(num_classes=num_classes)
    return model.to('cpu')

# 初始化SummaryWriter
writer = SummaryWriter('runs/bagging_experiment')

# Bagging
n_estimators = 5
model_types = ['shufflenet', 'mobilenetv2', 'mobilenetv3']
model_list = []
optimizers = []
schedulers = []

for _ in range(n_estimators):
    model_type = np.random.choice(model_types)
    model = create_model(model_type, num_classes=len(torch.unique(torch.tensor(train_labels))))
    model_list.append(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizers.append(optimizer)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # 每3个epoch学习率衰减为原来的0.1
    schedulers.append(scheduler)

criterion = nn.CrossEntropyLoss()

# 训练模型
for epoch in range(10):  # 训练10个epoch
    for model, optimizer, scheduler in zip(model_list, optimizers, schedulers):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to('cpu'), labels.to('cpu')

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 记录训练损失
            writer.add_scalar(f'Model_{model_list.index(model)}/Train Loss', loss.item(), epoch * len(train_loader) + train_loader.batch_size)

        # 更新学习率
        scheduler.step()

# 预测并评估模型
for model in model_list:
    model.eval()
y_pred = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to('cpu')
        outputs = torch.stack([model(images) for model in model_list]).mean(dim=0)
        y_pred.extend(outputs.argmax(dim=1).cpu().numpy())

# 计算准确率并记录
accuracy = accuracy_score(test_labels, y_pred)
writer.add_scalar('Test Accuracy', accuracy, epoch)

print("Bagging的准确率: ", accuracy)

# 关闭SummaryWriter
writer.close()