from torch.utils.tensorboard import SummaryWriter
from model import *
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='C:\/Users\JasonFeng\Desktop\python\CV\CIFAR10',
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root='C:\/Users\JasonFeng\Desktop\python\CV\CIFAR10',
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

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
jason = Jason()

# 创建函数
# 损失函数
loss_func = nn.CrossEntropyLoss()
# 优化器
learning_rate = 0.001  # 1e-3 = 0.001
optimizer = torch.optim.SGD(Jason().parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 5

# 添加tensorboard
writer = SummaryWriter(
    'C:\/Users\JasonFeng\Desktop\python\CV\CIFAR10/logs_train')

for i in range(epoch):
    print('-----------------epoch:', i, ' is training-----------------')

    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        outputs = jason(imgs)
        loss = loss_func(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print('train_step:', total_train_step, ' loss:', loss.item())
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = jason(imgs)
            loss = loss_func(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print('整体测试上的loss:', total_test_loss)
    print('整体测试上的accuracy:', total_accuracy/test_data_length)
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy/test_data_length, total_test_step)
    total_test_step = total_test_step + 1

    # 保存模型
    torch.save(jason, 'C:\/Users\JasonFeng\Desktop\python\CV\CIFAR10/model_train.pth')
    print('model saved')

writer.close()
