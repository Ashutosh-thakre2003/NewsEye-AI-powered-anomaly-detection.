import torch.nn as nn
import torchvision.models as models

class CNNModel(nn.Module):
    def __init__(self, num_classes=2, freeze_layers=True):
        super(CNNModel, self).__init__()
        # Load the ResNet50 model pre-trained on ImageNet
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Optionally freeze the early layers
        if freeze_layers:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Modify the final fully connected layer to match the number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

if __name__ == "__main__":
    model = CNNModel(num_classes=2, freeze_layers=True)  # Create the model instance with frozen layers
    print(model)
