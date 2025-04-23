import os
from flask import Flask, render_template, request, url_for
from wtforms import Form, FileField, validators
from PIL import Image
import torch
from torchvision import transforms, datasets

from preprocess import image_transform

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
FILLER_CLASSIFICATIONS = ("John Pokemon", 99.99)

cur_dir = os.path.dirname(__file__)
model = torch.load(os.path.join(cur_dir, 'model', 'best_model_fold1.pth'))


# class ImageForm(Form):
#     image = FileField('image')

def classify(image_path):
    # transform image so model can take it
    X = image_transform(image_path)
    with torch.no_grad():
        classification = model(X)
    y = model.predict_proba(X)
    return classification, y
    

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
        prediction, y = classify(filepath)
        return render_template('uploadedImage.html', 
                               filename=filename,
                               classification=prediction,
                               probability=y)
    else:
        return render_template('index.html', error="No file selected")
    
    
if __name__ == '__main__':
    app.run(debug=True)