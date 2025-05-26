import os
import torch
from torchvision import transforms
from PIL import Image
from django.shortcuts import render
from .forms import ImageUploadForm
from .model_def import HybridHandGestureModel, label_map
from django.core.files.storage import FileSystemStorage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'hybrid_hand_gesture_model.pth')

model = HybridHandGestureModel(num_classes=len(label_map))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_idx = predicted.item()
        for label, idx in label_map.items():
            if idx == predicted_idx:
                return label
    return "Unknown"

def index(request):
    result = None
    uploaded_image = None

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(image_file.name, image_file)
            uploaded_image = fs.url(filename)

            image = Image.open(image_file).convert('RGB')
            result = predict_image(image)
    else:
        form = ImageUploadForm()

    return render(request, 'index.html', {
        'form': form,
        'result': result,
        'uploaded_image': uploaded_image
    })
