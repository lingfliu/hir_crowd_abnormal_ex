import torch.nn.functional as F
import torch.nn as nn
import torch

""" convolution layer block"""
class Conv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, feature_size):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

"""visual transformer model"""
class Unet(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(Unet, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        feature_size = 8

        input_size = 2

        self.convD1 = Conv(input_size, feature_size)
        self.convD2 = Conv(feature_size, feature_size*2)
        self.convD3 = Conv(feature_size*2, feature_size*4)
        self.convD4 = Conv(feature_size*4, feature_size*8)
        self.convD5 = Conv(feature_size*8, feature_size*16)

        self.convU1 = Conv(feature_size*3, feature_size)
        self.convU2 = Conv(feature_size*6, feature_size*2)
        self.convU3 = Conv(feature_size*12, feature_size*4)
        self.convU4 = Conv(feature_size*24, feature_size*8)
        self.convU5 = Conv(feature_size*16, feature_size*16)


        self.convF1 = nn.Linear(in_features=feature_size*120*214, out_features=64)
        self.convF2 = nn.Linear(in_features=64, out_features=64)
        self.convF3 = nn.Linear(in_features=64, out_features=5)

        self.optim = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criteria = nn.CrossEntropyLoss()

        self.device = torch.device('cuda:0' if torch.cuda.device_count() >= 1 else 'cpu')


    def forward(self, x):
        xD1 = self.convD1(x)

        xD2 = nn.MaxPool2d(2)(xD1)
        xD2 = self.convD2(xD2)

        xD3 = nn.MaxPool2d(2)(xD2)
        xD3 = self.convD3(xD3)

        xD4 = nn.MaxPool2d(2)(xD3)
        xD4 = self.convD4(xD4)

        xD5 = nn.MaxPool2d(2)(xD4)
        xD5 = self.convD5(xD5)

        xU5 = self.convU5(xD5)

        xU4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(xU5)
        xU4 = nn.ConvTranspose2d(xU4.shape[1], xU4.shape[1], kernel_size=(2,1), stride=1).to(self.device)(xU4)
        xU4 = self.convU4(torch.cat([xU4, xD4], dim=1))

        xU3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(xU4)
        xU3 = nn.ConvTranspose2d(xU3.shape[1], xU3.shape[1], kernel_size=(1,2), stride=1).to(self.device)(xU3)
        xU3 = self.convU3(torch.cat([xU3, xD3], dim=1))

        xU2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(xU3)
        xU2 = nn.ConvTranspose2d(xU2.shape[1], xU2.shape[1], kernel_size=(1,2), stride=1).to(self.device)(xU2)
        xU2 = self.convU2(torch.cat([xU2, xD2], dim=1))

        xU1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(xU2)
        xU1 = self.convU1(torch.cat([xU1, xD1], dim=1))

        xU0 = nn.Flatten()(xU1)
        xU0 = self.convF1(xU0)
        xU0 = nn.ReLU(inplace=True)(xU0)
        xU0 = self.convF2(xU0)
        xU0 = nn.ReLU(inplace=True)(xU0)
        xU0 = self.convF3(xU0)
        xU0 = nn.ReLU(inplace=True)(xU0)

        o = F.log_softmax(xU0, dim=1)
        return o

    def loss(self, input, target):
        return self.criteria(input, target.argmax(1))

    def summarize(self):

        output = str(self)
        output += '\n'
        output += 'number of paramters {}'.format(self.net_size)
        return output