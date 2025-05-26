import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import json

# ✅ Define Model Class (Same as Training)
class HandGestureModel(nn.Module):
    def _init_(self, num_classes=18):
        super(HandGestureModel, self)._init_()
        self.base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Identity()
        self.lstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# ✅ Load Model
model = HandGestureModel(num_classes=18)

# ✅ Load Weights
try:
    model.load_state_dict(torch.load("hand_gesture_model.pth", map_location=torch.device("cpu")))
    model.eval()  # Set to evaluation mode
    print("✅ Model loaded successfully for real-time inference!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ✅ Load Gesture Labels
with open("label_mapping.json", "r") as f:
    gesture_labels = json.load(f)

# ✅ Define Image Transformations (Same as Training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Ensure consistency
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Must match training
])

# ✅ Prediction Function
def predict(frame, model):
    try:
        # Convert OpenCV frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = transform(image).unsqueeze(0)  # Add batch dimension

        model.eval()
        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)  # Apply softmax
            prediction_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction_idx].item()

        # ✅ Confidence threshold to filter uncertain predictions
        return gesture_labels[prediction_idx] if confidence > 0.3 else "Uncertain"

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return "Error"

# ✅ OpenCV Real-Time Webcam Capture
cap = cv2.VideoCapture(0)  # Open default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Resize for better display
    frame_resized = cv2.resize(frame, (400, 300))

    # ✅ Predict gesture on current frame
    gesture = predict(frame_resized, model)

    # ✅ Display Gesture Text
    cv2.putText(frame_resized, f"Gesture: {gesture}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ✅ Show Real-Time Video
    cv2.imshow("Real-Time Hand Gesture Recognition", frame_resized)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()