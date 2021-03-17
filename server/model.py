
import torch
from torch import nn
from torchvision.models import resnet34

class HotDogNotHotDogClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained_resnet34 = resnet34(pretrained=True)
        self.featureExtracter = nn.Sequential(pretrained_resnet34.conv1,
                                              pretrained_resnet34.bn1,
                                              pretrained_resnet34.relu,
                                              pretrained_resnet34.maxpool,
                                              pretrained_resnet34.layer1,
                                              pretrained_resnet34.layer2,
                                              pretrained_resnet34.layer3,
                                              pretrained_resnet34.layer4,
                                              pretrained_resnet34.avgpool)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.featureExtracter(x)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return self.sigmoid(x)
