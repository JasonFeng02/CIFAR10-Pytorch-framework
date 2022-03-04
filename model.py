import torch
import torch.nn as nn
import torch.nn.functional as F

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

if __name__ == '__main__':
    jason = Jason()
    input = torch.ones(64,3,32,32)
    output = jason(input)
    print(output.shape)