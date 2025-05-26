import torch
import torch.nn as nn
from transformers import SwinForImageClassification
from torchvision import models
from transformers import SwinConfig

label_map = {
    'call': 0, 'dislike': 1, 'fist': 2, 'four': 3, 'like': 4,
    'mute': 5, 'ok': 6, 'one': 7, 'palm': 8, 'peace': 9,
    'peace_inverted': 10, 'rock': 11, 'stop': 12, 'stop_inverted': 13,
    'three': 14, 'three2': 15, 'two_up': 16, 'two_up_inverted': 17
}

class HybridHandGestureModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridHandGestureModel, self).__init__()

        
        config = SwinConfig()
        self.swin_model = SwinForImageClassification(config)
        self.swin_model.classifier = nn.Identity()
        swin_feature_dim = 768

        self.resnet_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.resnet_model.fc = nn.Identity()
        resnet_feature_dim = 512

        combined_feature_dim = swin_feature_dim + resnet_feature_dim
        self.lstm = nn.LSTM(combined_feature_dim, 512, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        import torch.nn.functional as F
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        swin_features = self.swin_model(x).logits
        resnet_features = self.resnet_model(x)

        combined_features = torch.cat((swin_features, resnet_features), dim=1)
        combined_features = combined_features.unsqueeze(1)

        lstm_output, _ = self.lstm(combined_features)
        output = self.fc(lstm_output[:, -1, :])

        return output
