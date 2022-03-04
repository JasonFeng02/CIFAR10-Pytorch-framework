
# from model import *
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

# 准备数据集
import time
train_data = torchvision.datasets.CIFAR10(root='.\CIFAR10',
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root='.\CIFAR10',
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)


# length 数据集长度
train_data_length = len(train_data)
test_data_length = len(test_data)
# 验证数据集长度
print('train_data_length:', train_data_length)
print('test_data_length:', test_data_length)

# 利用dataloader加载数据

train_dataloader = DataLoader(train_data, batch_size=32)
test_dataloader = DataLoader(test_data, batch_size=32)

# 创建网络模型
class Jason(nn.Module):
    def __init__(self):
        super().__init__().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.model(x)
        return x

jason = Jason()
if torch.cuda.is_available():
    jason = jason.cuda()

# 创建函数
# 损失函数
loss_func = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_func = loss_func.cuda()
# 优化器
learning_rate = 0.01  # 1e-3 = 0.001
optimizer = torch.optim.SGD(Jason().parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 2

start_time = time.time()

for i in range(epoch):
    print('-----------------epoch:', i+1, ' is training-----------------')

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = jason(imgs)
        loss = loss_func(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print('train_step:', total_train_step, ' loss:', loss.item())

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = jason(imgs)
            loss = loss_func(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print('整体测试上的loss:{}'.format(total_test_loss))
    print('整体测试上的accuracy:{}'.format(total_accuracy/test_data_length))

    # 保存模型
    torch.save(jason, '.\CV\CIFAR10/model_train.pth')
    torch.save(jason.state_dict(), '.\CV\CIFAR10/model_train_state_dict.pth')

