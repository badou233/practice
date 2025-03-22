import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
from Dataset import LeafDataset  # 导入 LeafDataset 类

##读取数据
train_data = pd.read_csv('classify-leaves/train.csv')
test_data = pd.read_csv('classify-leaves/test.csv')

print(train_data.shape)
print(test_data.shape)

# 按照8:2的比例划分为训练集和验证集
train_set, val_set = train_test_split(train_data, test_size=0.2, random_state=42)

# # 打印划分后的数据集形状
# print("训练集形状:", train_set.shape)
# print("验证集形状:", val_set.shape)
# print(train_set.head())


## 数据预处理

# 创建标签映射字典
unique_labels = train_data['label'].unique()
unique_labels = np.sort(unique_labels)
label_to_index = {label: index for index, label in enumerate(unique_labels)}
index_to_label = {index: label for label, index in label_to_index.items()}

# 对训练集进行图像翻转，测试集不用
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomVerticalFlip(),
     torchvision.transforms.ToTensor()])
trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# 图片所在的根目录
root_dir = "classify-leaves"

train_dataset = LeafDataset(train_set, root_dir, train_augs, label_to_index)
val_dataset = LeafDataset(val_set, root_dir, trans, label_to_index)
test_dataset = LeafDataset(test_data, root_dir, trans, label_to_index)

# 创建 DataLoader 对象
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=d2l.get_dataloader_workers())
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=d2l.get_dataloader_workers())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=d2l.get_dataloader_workers())


# # 遍历 DataLoader 获取第一批数据
# for images, labels in train_loader:
#     print(images.size())
#     print(labels)
#     break


##构建ResNet模型
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    

b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(input_channels, num_channels, num_residuals,
                first_block=False):
    blk = []

    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
        return blk


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 176))




lr, num_epochs = 0.01, 5
d2l.train_ch6(net, train_loader, val_loader, num_epochs, lr, d2l.try_gpu())

# 保存模型状态字典
torch.save(net.state_dict(), "net_{}_{}_{}.pth".format(lr, batch_size, num_epochs))
print("模型状态字典已保存")


# # 加载模型状态字典的代码
# net = nn.Sequential(b1, b2, b3, b4, b5,
#                     nn.AdaptiveAvgPool2d((1,1)),
#                     nn.Flatten(), nn.Linear(512, 176))
# net.load_state_dict(torch.load("net_{}_{}_{}.pth".format(lr, batch_size, num_epochs)))
# print("模型状态字典已加载")


# 进行预测并输出测试结果
predictions = []
image_names = []
with torch.no_grad():
    for images,names in test_loader:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = images.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(predicted)):
            label = index_to_label[predicted[i].item()]
            predictions.append(label)
            image_names.append(names[i])

# 保存预测结果到 CSV 文件
result_df = pd.DataFrame({
    'image': image_names,
    'label': predictions
})
result_df.to_csv('测试结果2.csv', index=False)

