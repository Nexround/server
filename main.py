from flask import Flask, request, send_from_directory
from flask_cors import CORS
from F.utils import *
import torch 
import torchvision.transforms as transforms
from PIL import Image
import os
# from model.anime_gan import Generator
from networks import *

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload'
app.config['RESULT_FOLDER'] = 'result'
CORS(app)
# model = Generator()
# model.load_state_dict(torch.load('model/model.pth', map_location='cpu'))
model = torch.load('model/model.pth', map_location='cpu')
model.eval()
def inference(image_path):
    
    image_tensor = transform(Image.open(image_path))
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
    result_image = tensor2im(output)
    result_image = transforms.ToPILImage()(result_image)


    return result_image
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    file = request.files['image']
    filename = file.filename
    image = Image.open(file.stream).convert('RGB')
    image_path = 'upload/' + filename
    image.save(image_path)
    result_image = inference(image_path)
    result_name = os.path.splitext(filename)[0] + '.jpg'
    result_path = 'result/' + result_name
    result_image.save(result_path)
    return send_from_directory(app.config['RESULT_FOLDER'], result_name)
@app.route('/result/<image_name>', methods=['GET'])
def result(image_name):
    result_name = os.path.splitext(image_name)[0] + '.jpg'
    return send_from_directory(app.config['RESULT_FOLDER'], result_name, mimetype='image/jpg')
if __name__ == '__main__':
    app.run(debug=True, port=5000)