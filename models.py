## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # input WXH 96X96, after convolution (96-4)/1 + 1 = 93X93; pooling -> 46X46
        # input WXH 46X46, after convolution (46-3)/1 + 1 = 44X44; pooling -> 22X22
        # input WXH 22X22, after convolution (22-2)/1 + 1 = 21X21; pooling -> 10X10
        # input WXH 10X10, after convolution (10-1)/1 + 1 = 10X10; pooling -> 5X5
        
        # input WXH 224X224, after convolution (224 - 4)/1 + 1; pooling -> 110x110 
        # input WXH 110X110, after convolution (110 - 3)/1 + 1; pooling -> 54x54 
        # input WXH 224X224, after convolution (54 - 2)/1 + 1; pooling -> 26x26 
        # input WXH 224X224, after convolution (26 - 1)/1 + 1; pooling -> 13x13 
        
        # Convlution layers 
        self.conv1 = nn.Conv2d(1, 32, 4)                 
        self.conv2 = nn.Conv2d(32, 64, 3)                
        self.conv3 = nn.Conv2d(64, 128, 2)               
        self.conv4 = nn.Conv2d(128, 256, 1)              
        
        # Maxpool
        self.pool = nn.MaxPool2d(2,2)
        
        # fully connected layers
        self.fc1 = nn.Linear(256*13*13,1000 )
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
        # dropouts
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.6)
        
        # initializing weights
        # initialize convolutional layers to random weight from uniform distribution
        
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            I.uniform_(conv.weight)
        
        # initialize fully connected layers weight using Glorot uniform initialization
        for fc in [self.fc1, self.fc2, self.fc3]:
            I.xavier_uniform_(fc.weight, gain=I.calculate_gain('relu'))
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # normalize
        self.norm = nn.BatchNorm1d(136)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        ## Full Network
        ## Input -> conv1 -> Activation1 -> pool -> drop1 -> conv2 -> Activation2 -> pool -> drop2
        ##  -> conv3 -> Activation3 -> pool -> drop3 -> conv4 -> Activation4 -> pool -> drop4 ->
        ## flatten -> fc1 -> Activation5 -> drop5 -> fc2 -> Activation6 -> drop6 -> fc3
        
        ## Activation1 to Activation5 are Exponential Linear Units
        ## Activation6 is Linear Activation Function
        x = self.pool(F.elu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool(F.elu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool(F.elu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool(F.elu(self.conv4(x)))
        x = self.drop4(x)
        x = x.view(x.size()[0], -1)
        x = F.elu(self.fc1(x))
        x = self.drop5(x)
        x = F.relu(self.fc2(x))
        x = self.drop6(x)
        x = self.fc3(x)
        x = self.norm(x)
        
        return x
