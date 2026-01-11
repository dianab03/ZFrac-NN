import torch
import torch.nn as nn
import torchvision.models as models


# Constants for model architecture
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_DROPOUT_RATE = 0.3
HIDDEN_SIZE_REDUCTION_FACTOR = 2  # Second layer is half the size of first


class ZFracNN(nn.Module):
    """
    Neural network for classifying fractal (zonal fractal) features.
    
    This is a simple feedforward network designed to work with pre-extracted
    fractal features rather than raw images. It uses dropout for regularization.
    """
    
    def __init__(self, input_dim, num_classes, hidden=DEFAULT_HIDDEN_SIZE):
        """
        Initialize the fractal feature classifier.
        
        Args:
            input_dim: Number of input features (from fractal feature extraction)
            num_classes: Number of output classes
            hidden: Size of the first hidden layer
        """
        super().__init__()
        second_hidden_size = hidden // HIDDEN_SIZE_REDUCTION_FACTOR
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(DEFAULT_DROPOUT_RATE),
            nn.Linear(hidden, second_hidden_size),
            nn.ReLU(),
            nn.Dropout(DEFAULT_DROPOUT_RATE),
            nn.Linear(second_hidden_size, num_classes)
        )
    
    def forward(self, features):
        """
        Forward pass through the network.
        
        Args:
            features: Input tensor of fractal features
        
        Returns:
            Logits for each class
        """
        return self.network(features)


# Constants for CNN architectures
VGG16_FEATURE_DIM = 4096
RESNET18_FEATURE_DIM = 512
DENSENET121_FEATURE_DIM = 1024


class CNN(nn.Module):
    """
    Convolutional neural network using pretrained backbones.
    
    This class wraps popular CNN architectures (ResNet, VGG, DenseNet) and
    replaces their final classification layer to match the number of classes
    in our dataset.
    """
    
    def __init__(self, num_classes, backbone='resnet18', pretrained=True):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of output classes
            backbone: Architecture to use ('resnet18', 'vgg16', or 'densenet121')
            pretrained: If True, use ImageNet pretrained weights
        """
        super().__init__()
        self.backbone_name = backbone
        
        if backbone == 'vgg16':
            weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.vgg16(weights=weights)
            # Replace the final classification layer
            self.model.classifier[6] = nn.Linear(VGG16_FEATURE_DIM, num_classes)
            
        elif backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet18(weights=weights)
            # Replace the final fully connected layer
            self.model.fc = nn.Linear(RESNET18_FEATURE_DIM, num_classes)
            
        elif backbone == 'densenet121':
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.densenet121(weights=weights)
            # Replace the final classification layer
            self.model.classifier = nn.Linear(DENSENET121_FEATURE_DIM, num_classes)
            
        else:
            raise ValueError(f"Unknown backbone architecture: {backbone}")
    
    def forward(self, images):
        """
        Forward pass through the CNN.
        
        Args:
            images: Input tensor of images
        
        Returns:
            Logits for each class
        """
        return self.model(images)
    
    def get_layer_features(self, images, layer_index):
        """
        Extract intermediate layer features from the CNN.
        
        This is used for analysis purposes (e.g., CCA/CKA) to compare
        CNN representations with fractal features.
        
        Args:
            images: Input tensor of images
            layer_index: Index of the layer to extract features from
        
        Returns:
            Feature tensor from the specified layer
        """
        if self.backbone_name == 'resnet18':
            # ResNet18 forward pass through layers
            feature_tensor = self.model.conv1(images)
            feature_tensor = self.model.bn1(feature_tensor)
            feature_tensor = self.model.relu(feature_tensor)
            feature_tensor = self.model.maxpool(feature_tensor)
            
            # ResNet has 4 main residual blocks
            residual_blocks = [
                self.model.layer1, 
                self.model.layer2, 
                self.model.layer3, 
                self.model.layer4
            ]
            
            for block_index, residual_block in enumerate(residual_blocks):
                feature_tensor = residual_block(feature_tensor)
                if block_index == layer_index:
                    return feature_tensor
            
            # If layer_index is beyond the blocks, return final pooled features
            feature_tensor = self.model.avgpool(feature_tensor)
            return torch.flatten(feature_tensor, 1)
            
        elif self.backbone_name == 'vgg16':
            # VGG16 extracts features from the convolutional layers
            feature_tensor = self.model.features(images)
            if layer_index < len(self.model.features):
                return feature_tensor
            
            # If beyond feature layers, return final pooled features
            feature_tensor = self.model.avgpool(feature_tensor)
            feature_tensor = torch.flatten(feature_tensor, 1)
            return feature_tensor
            
        elif self.backbone_name == 'densenet121':
            # DenseNet extracts features from the dense blocks
            feature_tensor = self.model.features(images)
            feature_tensor = nn.functional.relu(feature_tensor, inplace=True)
            feature_tensor = nn.functional.adaptive_avg_pool2d(feature_tensor, (1, 1))
            feature_tensor = torch.flatten(feature_tensor, 1)
            return feature_tensor
        else:
            # Fallback for other architectures
            return self.model.features(images)


# Constants for SimpleCNN architecture
SIMPLE_CNN_INPUT_CHANNELS = 3
SIMPLE_CNN_FIRST_CHANNELS = 32
SIMPLE_CNN_CHANNEL_MULTIPLIER = 2  # Each layer doubles channels
SIMPLE_CNN_KERNEL_SIZE = 3
SIMPLE_CNN_PADDING = 1
SIMPLE_CNN_POOL_SIZE = 2
SIMPLE_CNN_FINAL_POOL_SIZE = 4
SIMPLE_CNN_CLASSIFIER_HIDDEN = 512
SIMPLE_CNN_DROPOUT_RATE = 0.5


class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network from scratch.
    
    This is a basic CNN architecture without pretrained weights, useful
    for comparison with pretrained models or when pretrained models aren't available.
    """
    
    def __init__(self, num_classes):
        """
        Initialize the simple CNN.
        
        Args:
            num_classes: Number of output classes
        """
        super().__init__()
        
        # Feature extraction layers: progressively increase channels
        channel_sequence = [
            SIMPLE_CNN_FIRST_CHANNELS,
            SIMPLE_CNN_FIRST_CHANNELS * SIMPLE_CNN_CHANNEL_MULTIPLIER,
            SIMPLE_CNN_FIRST_CHANNELS * SIMPLE_CNN_CHANNEL_MULTIPLIER ** 2,
            SIMPLE_CNN_FIRST_CHANNELS * SIMPLE_CNN_CHANNEL_MULTIPLIER ** 3
        ]
        
        self.features = nn.Sequential(
            nn.Conv2d(SIMPLE_CNN_INPUT_CHANNELS, channel_sequence[0], 
                     SIMPLE_CNN_KERNEL_SIZE, padding=SIMPLE_CNN_PADDING),
            nn.BatchNorm2d(channel_sequence[0]),
            nn.ReLU(),
            nn.MaxPool2d(SIMPLE_CNN_POOL_SIZE),
            
            nn.Conv2d(channel_sequence[0], channel_sequence[1], 
                     SIMPLE_CNN_KERNEL_SIZE, padding=SIMPLE_CNN_PADDING),
            nn.BatchNorm2d(channel_sequence[1]),
            nn.ReLU(),
            nn.MaxPool2d(SIMPLE_CNN_POOL_SIZE),
            
            nn.Conv2d(channel_sequence[1], channel_sequence[2], 
                     SIMPLE_CNN_KERNEL_SIZE, padding=SIMPLE_CNN_PADDING),
            nn.BatchNorm2d(channel_sequence[2]),
            nn.ReLU(),
            nn.MaxPool2d(SIMPLE_CNN_POOL_SIZE),
            
            nn.Conv2d(channel_sequence[2], channel_sequence[3], 
                     SIMPLE_CNN_KERNEL_SIZE, padding=SIMPLE_CNN_PADDING),
            nn.BatchNorm2d(channel_sequence[3]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((SIMPLE_CNN_FINAL_POOL_SIZE, SIMPLE_CNN_FINAL_POOL_SIZE))
        )
        
        # Classification layers
        final_channels = channel_sequence[3]
        flattened_size = final_channels * SIMPLE_CNN_FINAL_POOL_SIZE * SIMPLE_CNN_FINAL_POOL_SIZE
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, SIMPLE_CNN_CLASSIFIER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(SIMPLE_CNN_DROPOUT_RATE),
            nn.Linear(SIMPLE_CNN_CLASSIFIER_HIDDEN, num_classes)
        )
    
    def forward(self, images):
        """
        Forward pass through the network.
        
        Args:
            images: Input tensor of images
        
        Returns:
            Logits for each class
        """
        feature_tensor = self.features(images)
        return self.classifier(feature_tensor)
    
    def get_layer_features(self, images, layer_index):
        """
        Extract intermediate layer features.
        
        Args:
            images: Input tensor of images
            layer_index: Index of the layer to extract features from
        
        Returns:
            Feature tensor from the specified layer
        """
        feature_tensor = images
        # Each "layer" in our indexing corresponds to 4 sequential operations:
        # Conv2d, BatchNorm, ReLU, MaxPool
        target_layer_index = layer_index * 4 + 3
        
        for current_index, layer in enumerate(self.features):
            feature_tensor = layer(feature_tensor)
            if current_index == target_layer_index:
                return feature_tensor
        
        return feature_tensor
