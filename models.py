import torch
import torch.nn as nn
import torchvision.models as models


class ZFracNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, num_classes, backbone='resnet18', pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        
        if backbone == 'vgg16':
            w = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.vgg16(weights=w)
            self.model.classifier[6] = nn.Linear(4096, num_classes)
        elif backbone == 'resnet18':
            w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet18(weights=w)
            self.model.fc = nn.Linear(512, num_classes)
        else:
            raise ValueError(f"unknown backbone {backbone}")
    
    def forward(self, x):
        return self.model(x)
    
    def get_layer_features(self, x, layer_idx):
        if self.backbone_name == 'resnet18':
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            
            layers = [self.model.layer1, self.model.layer2, 
                      self.model.layer3, self.model.layer4]
            
            for i, layer in enumerate(layers):
                x = layer(x)
                if i == layer_idx:
                    return x
            
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)
        else:
            return self.model.features(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
    def get_layer_features(self, x, layer_idx):
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == layer_idx * 4 + 3:
                return x
        return x
