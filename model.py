import torch
import torch.nn as nn
from torchvision import models

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''
        input shape : (3, 64, 64)

        '''
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 75)

        self.net = model_ft

        '''
        output shape : (75,)

        '''

    # define the feedforward behavior
    def forward(self, x):
        x = self.net(x)

        return x.view(x.size(0), -1)
