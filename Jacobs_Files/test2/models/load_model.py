import torch
from timm import create_model
 
def get_model():
    model = create_model('convnext_base', pretrained=True, num_classes=149)
    model.load_state_dict(torch.load('best_model_fold1.pth'), map_location='cpu')
    model.eval()
    return model


def predict(model, image):
    pass

def predict_proba(image):
    pass