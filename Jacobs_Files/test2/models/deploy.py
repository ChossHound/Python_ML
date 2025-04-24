#!/usr/bin/env python
# coding: utf-8

# In[10]:


import timm
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# In[11]:


# Import Data Metadata
with open("class_names.json", "r") as f:
  class_names = json.load(f)

# print(class_names)


# In[12]:


# Settings
MODEL_NAME = "convnext_base"
NUM_CLASSES = len(class_names)  # Set based on the number of class names
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_WEIGHT_PATH = "best_model_fold1.pth"  # Path to the model weights


# In[13]:


# Rebuild model and load weights
def get_model():
  model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
  model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE))
  model.to(DEVICE)
  # model.eval()
  return model


# In[14]:


# Define transform (same as eval_transform)
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# In[15]:


def load_and_transform_image(img_path, transform, device):
  """
  Load an image from the given path, apply the specified transform, and move it to the specified device.

  Args:
    img_path (str): Path to the image file.
    transform (torchvision.transforms.Compose): Transformations to apply to the image.
    device (torch.device): Device to move the image tensor to.

  Returns:
    torch.Tensor: Transformed image tensor with an added batch dimension.
  """
  image = Image.open(img_path).convert("RGB")
  return transform(image).unsqueeze(0).to(device)
  # return transform(image)

# # Example usage
# img_path = "upload/gengar-gen1.jpg"  # Change to your image path
# image_tensor = load_and_transform_image(img_path, eval_transform, DEVICE)


# In[16]:


# with torch.no_grad():
#     output = model(image_tensor)
#     predicted_class = torch.argmax(output, dim=1).item()

# print(f"Predicted class index: {predicted_class}")
# print(f"Predicted class name: {class_names[predicted_class]}")


# In[17]:


# import json

# # Export class_names to a JSON file
# with open("class_names.json", "w") as f:
#   json.dump(class_names, f)

