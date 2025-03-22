import numpy as np
import pandas as pd
import torch
import torchvision
from IPython import display
from matplotlib import pyplot as plt
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
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=d2l.get_dataloader_workers())
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=d2l.get_dataloader_workers())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=d2l.get_dataloader_workers())


# # 遍历 DataLoader 获取第一批数据
# for images, labels in train_loader:
#     print(images.size())
#     print(labels)
#     break


##构建ResNet-18模型
def train_batch_ch13(net, X, y, loss, trainer, devices):
    X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """用多GPU进行模型训练"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = net.to(devices[0])

    for epoch in range(num_epochs):
        print("-------第 {} 轮训练开始-------".format(epoch + 1))
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print("训练次数：{}, Loss: {}".format(i, l.item()))
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
            torch.cuda.empty_cache()  # 释放显存

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    # 显示animator记录的图像
    display.display(animator.fig)
    plt.show()
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')

devices, net = d2l.try_all_gpus(), d2l.resnet18(176, 3)
net = net.to(devices[0])

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

lr, num_epochs = 0.01, 10
loss = nn.CrossEntropyLoss(reduction="none")
trainer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch13(net, train_loader, val_loader, loss, trainer, num_epochs, devices)


# 保存模型状态字典
torch.save(net.state_dict(), "net_{}_{}_{}_resnet18.pth".format(lr, batch_size, num_epochs))
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
    for images, names in test_loader:
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
result_df.to_csv('测试结果3.csv', index=False)
