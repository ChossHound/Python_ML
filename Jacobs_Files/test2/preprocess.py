from torchvision import transforms, datasets
from PIL import Image


IMG_SIZE = 224

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
    
def image_transform(image_path):
    image = Image.open(image_path).convert('RGB')
    image = eval_transform(image) #.unsqueeze(0).to(device)
    return image
