import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable

class AdversarialLayer(torch.autograd.Function):
  def __init__(self, high_value=1.0):
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = high_value
    self.max_iter = 10000.0
    
  def forward(self, input):
    self.iter_num += 1
    return input * 1.0

  def backward(self, gradOutput):
    coeff = 2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha*self.iter_num / self.max_iter)) - (self.high - self.low) + self.low
    return -coeff * gradOutput

class BackAdversarialLayer(torch.autograd.Function):
  def __init__(self):
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0
    
  def forward(self, input):
    self.iter_num += 1
    return input * 1.0

  def backward(self, gradOutput):
    coeff = 2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha*self.iter_num / self.max_iter)) - (self.high - self.low) + self.low
    return coeff * gradOutput

class RMANLayer(torch.autograd.Function):
  def __init__(self, input_dim_list=[], output_dim=1024):
    self.input_num = len(input_dim_list)
    self.output_dim = output_dim
    self.random_matrix = [Variable(torch.randn(input_dim_list[i], output_dim)) for i in xrange(self.input_num)]
    for val in self.random_matrix:
      val.requires_grad = False
  def forward(self, input_list):
    return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in xrange(self.input_num)]
    return_list[0] = return_list[0] / float(self.output_dim)
    return return_list
  def cuda(self):
    self.random_matrix = [val.cuda() for val in self.random_matrix]

class SilenceLayer(torch.autograd.Function):
  def __init__(self):
    pass
  def forward(self, input):
    return input * 1.0

  def backward(self, gradOutput):
    return 0 * gradOutput


# convnet without the last layer
class AlexNetFc(nn.Module):
  def __init__(self):
    super(AlexNetFc, self).__init__()
    model_alexnet = models.alexnet(pretrained=True)
    self.features = model_alexnet.features
    self.classifier = nn.Sequential()
    for i in xrange(6):
      self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
    self.__in_features = model_alexnet.classifier[6].in_features
  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256*6*6)
    x = self.classifier(x)
    return x

  def output_num(self):
    return self.__in_features

class ResNet18Fc(nn.Module):
  def __init__(self):
    super(ResNet18Fc, self).__init__()
    model_resnet18 = models.resnet18(pretrained=True)
    self.conv1 = model_resnet18.conv1
    self.bn1 = model_resnet18.bn1
    self.relu = model_resnet18.relu
    self.maxpool = model_resnet18.maxpool
    self.layer1 = model_resnet18.layer1
    self.layer2 = model_resnet18.layer2
    self.layer3 = model_resnet18.layer3
    self.layer4 = model_resnet18.layer4
    self.avgpool = model_resnet18.avgpool
    self.__in_features = model_resnet18.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features

class ResNet34Fc(nn.Module):
  def __init__(self):
    super(ResNet34Fc, self).__init__()
    model_resnet34 = models.resnet34(pretrained=True)
    self.conv1 = model_resnet34.conv1
    self.bn1 = model_resnet34.bn1
    self.relu = model_resnet34.relu
    self.maxpool = model_resnet34.maxpool
    self.layer1 = model_resnet34.layer1
    self.layer2 = model_resnet34.layer2
    self.layer3 = model_resnet34.layer3
    self.layer4 = model_resnet34.layer4
    self.avgpool = model_resnet34.avgpool
    self.__in_features = model_resnet34.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features

class ResNet50Fc(nn.Module):
  def __init__(self):
    super(ResNet50Fc, self).__init__()
    model_resnet50 = models.resnet50(pretrained=True)
    self.conv1 = model_resnet50.conv1
    self.bn1 = model_resnet50.bn1
    self.relu = model_resnet50.relu
    self.maxpool = model_resnet50.maxpool
    self.layer1 = model_resnet50.layer1
    self.layer2 = model_resnet50.layer2
    self.layer3 = model_resnet50.layer3
    self.layer4 = model_resnet50.layer4
    self.avgpool = model_resnet50.avgpool
    self.__in_features = model_resnet50.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features

class ResNet101Fc(nn.Module):
  def __init__(self):
    super(ResNet101Fc, self).__init__()
    model_resnet101 = models.resnet101(pretrained=True)
    self.conv1 = model_resnet101.conv1
    self.bn1 = model_resnet101.bn1
    self.relu = model_resnet101.relu
    self.maxpool = model_resnet101.maxpool
    self.layer1 = model_resnet101.layer1
    self.layer2 = model_resnet101.layer2
    self.layer3 = model_resnet101.layer3
    self.layer4 = model_resnet101.layer4
    self.avgpool = model_resnet101.avgpool
    self.__in_features = model_resnet101.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features


class ResNet152Fc(nn.Module):
  def __init__(self):
    super(ResNet152Fc, self).__init__()
    model_resnet152 = models.resnet152(pretrained=True)
    self.conv1 = model_resnet152.conv1
    self.bn1 = model_resnet152.bn1
    self.relu = model_resnet152.relu
    self.maxpool = model_resnet152.maxpool
    self.layer1 = model_resnet152.layer1
    self.layer2 = model_resnet152.layer2
    self.layer3 = model_resnet152.layer3
    self.layer4 = model_resnet152.layer4
    self.avgpool = model_resnet152.avgpool
    self.__in_features = model_resnet152.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features

network_dict = {"AlexNet":AlexNetFc, "ResNet18":ResNet18Fc, "ResNet34":ResNet34Fc, "ResNet50":ResNet50Fc, "ResNet101":ResNet101Fc, "ResNet152":ResNet152Fc}
