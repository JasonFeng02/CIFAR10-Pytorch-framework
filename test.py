import torchvision
import torch
from PIL import Image
import torch.nn as nn

#加载图片
Image_path = './CV/CIFAR10/test pic/dog.png'
image = Image.open(Image_path)
print(image)

#图片预处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

#加载网络模型
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

model = torch.load('./CV/CIFAR10/model_train.pth')
#print(model)
image = torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)