import os
from flask import Flask, render_template, request, url_for
from wtforms import Form, FileField, validators
from PIL import Image
import torch
from torch.nn.functional import softmax as softmax
from torchvision import transforms, datasets
import timm
from models.deploy import load_and_transform_image, eval_transform, class_names
from models.data import compute_hash, image_exists_in_dataset, get_all_image_paths
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
FILLER_CLASSIFICATIONS = ("John Pokemon", 99.99)
NUM_CLASSES = 149
DEVICE = torch.device('cpu')

cur_dir = os.path.dirname(__file__)
model = timm.create_model('convnext_base', pretrained=True, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('models/best_model_fold1.pth', map_location=torch.device('cpu')))
model.eval()

with open('image_hashes.pkl', 'rb') as file:
    hashes = pickle.load(file)


def classify(image_path):
    # transform image so model can take it
    X = load_and_transform_image(image_path, eval_transform, DEVICE)
    with torch.no_grad():
        output = model(X)
        probabilities = softmax(output, dim=1)
        y = output.argmax(dim=1).item()
        pred_class_proba = probabilities[0][y].item()
    return y, pred_class_proba

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image = request.files['image']
    if image and image.filename:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(filepath)
        filename = image.filename
        y, proba = classify(filepath)
        in_train_set = image_exists_in_dataset(filepath, hashes)
        predicted_img_path = 'reference/' + class_names[y] + '.jpg'
        # return render_template('uploadedImage.html', filename=filename)
        return render_template('uploadedImage.html', 
                               filename=filename,
                               classification=class_names[y],
                               probability=round(proba * 100, 2),
                               in_set= in_train_set,
                               reference_filename = predicted_img_path)
    # else:
        return render_template('index.html', error="No file selected")
    
    
if __name__ == '__main__':
    app.run(debug=True)