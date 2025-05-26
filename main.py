import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from transformers import SwinForImageClassification

# Define Paths
DATASET_PATH = r"D:\123\hagrid-sample-120k-384p\Hand Gesture Recognition\hagrid_120k"
ANNOTATIONS_DIR = r"D:\123\hagrid-sample-120k-384p\Hand Gesture Recognition\ann_train_val"

# Load Annotations
def load_annotations(annotations_dir):
    annotations = []
    for file in os.listdir(annotations_dir):
        if file.endswith(".json"):
            gesture_name = file.replace(".json", "")
            file_path = os.path.join(annotations_dir, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                for image_id in data.keys():
                    image_filename = f"{image_id}.jpg"
                    annotations.append((image_filename, gesture_name))
    return annotations

annotations = load_annotations(ANNOTATIONS_DIR)

# Extract Labels
unique_labels = sorted(set(label for _, label in annotations))
label_map = {label: idx for idx, label in enumerate(unique_labels)}

# Define Dataset Class
class HandGestureDataset(Dataset):
    def _init_(self, root_dir, annotations, transform=None):
        self.root_dir = root_dir
        self.annotations = [
            item for item in annotations if os.path.exists(os.path.join(root_dir, f"train_val_{item[1]}", item[0]))
        ]
        self.transform = transform

    def _len_(self):
        return len(self.annotations)

    def _getitem_(self, idx):
        img_name, label_name = self.annotations[idx]
        img_path = os.path.join(self.root_dir, f"train_val_{label_name}", img_name)
        image = Image.open(img_path).convert('RGB')
        label = label_map[label_name]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load Dataset
dataset = HandGestureDataset(DATASET_PATH, annotations, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

# Define Hybrid Model
class HybridHandGestureModel(nn.Module):
    def _init_(self, num_classes):
        super(HybridHandGestureModel, self)._init_()
        
        # Swin Transformer 
        self.swin_model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.swin_model.classifier = nn.Identity()
        swin_feature_dim = 768  
        
        # ResNet34 
        self.resnet_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.resnet_model.fc = nn.Identity()
        resnet_feature_dim = 512  
        
        # LSTM Layer
        combined_feature_dim = swin_feature_dim + resnet_feature_dim
        self.lstm = nn.LSTM(combined_feature_dim, 512, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        # Ensure input is resized to 224x224 for Swin Transformer
        x = nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        
        swin_features = self.swin_model(x).logits
        resnet_features = self.resnet_model(x)
        
        # Combine features
        combined_features = torch.cat((swin_features, resnet_features), dim=1)
        combined_features = combined_features.unsqueeze(1)
        
        # Pass through BiLSTM
        lstm_output, _ = self.lstm(combined_features)
        output = self.fc(lstm_output[:, -1, :])
        
        return output

# Training Function
def train_model(model, train_loader, test_loader, num_epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        correct, total, total_loss = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Train Accuracy={train_acc:.2f}%")
    return model

# Run Training
if _name_ == '_main_':
    num_classes = len(unique_labels)
    print("\nTraining Hybrid Hand Gesture Model...")
    model = HybridHandGestureModel(num_classes)
    model = train_model(model, train_loader, test_loader, num_epochs=10, lr=0.0001)
    
    # Save Full Model
    torch.save(model, "hybrid_hand_gesture_model_full.pth")
    print("✅ Full model saved successfully as 'hybrid_hand_gesture_model_full.pth'.")
    
    # Save Model State Dictionary
    torch.save(model.state_dict(), "hybrid_hand_gesture_model.pth")
    print("✅ Model state_dict saved successfully as 'hybrid_hand_gesture_model.pth'.")