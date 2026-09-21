import torch 
import torch.nn as nn
import torch.nn.functional as F

class My_DNN(nn.Module):
    def __init__(self, num_classes = 4, in_channels = 3):
        super(My_DNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size = 3, padding = 1)
        self.batch1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 128, kernel_size = 3, padding = 1)
        self.batch2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.batch3 = nn.BatchNorm2d(128)

        self.skip = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout = nn.Dropout(p = 0.3)
        self.global_p = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.pool(x)  
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        residual = x
        x = self.pool(x) 
        x = self.conv3(x)
        x = self.batch3(x)
        residual = self.pool(residual)
        residual = self.skip(residual)
        x = x + residual 
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)
        x = self.global_p(x)  
        x = torch.flatten(x, 1)  
        x = self.fc(x)
        return x
