from flask import Flask, render_template, request, jsonify
from scipy.misc import imsave, imread, imresize, imshow
import numpy as np
import re
from CapsNet import capsules
import torch
from torchvision import transforms
import base64
from CNN import Net
from skimage import io

device = torch.device("cpu")
cuda = False

A, B, C, D = 64, 8, 16, 16
# add cnn model as well
capsules_model = capsules(A=A, B=B, C=C, D=D, E=10,
                          iters=2, cuda=False).to(device)
capsules_model.load_state_dict(torch.load('./saved_model/mnist_capsules.pth'))
capsules_model.eval()
cnn_model = Net()
cnn_model.load_state_dict(torch.load('./saved_model/mnist_cnn.pth'))

app = Flask(__name__)


def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)', imgData1).group(1)
    # print(imgstr)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


@app.route('/', methods=['GET', 'POST'])
def basic():

    return render_template('index.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():

    imgData = request.get_data()

    convertImage(imgData)
    x = imread('output.png',mode='L')
    print(x.shape)
    x = np.invert(x)
    x = imresize(x, (28, 28))
    x = x.astype('float32')
    x = x.reshape((1, 1, 28, 28))
    x = torch.from_numpy(x)

    # perform the prediction
    caps_output = capsules_model(x)
    caps_output = caps_output.detach().numpy()

    cnn_output = cnn_model(x)
    cnn_output = cnn_output.detach().numpy()

    print(caps_output)
    print(cnn_output)
    # convert the response to a string
    response = "Capsules output :"+np.array_str(np.argmax(
        caps_output, axis=1))+" CNN output :"+np.array_str(np.argmax(cnn_output, axis=1))
    return response


if __name__ == '__main__':
    app.run(threaded=True, debug=True, host='0.0.0.0')
