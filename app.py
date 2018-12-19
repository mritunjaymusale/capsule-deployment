from flask import Flask, render_template, request
from scipy.misc import  imread, imresize
import numpy as np
import re
from CapsNet import capsules
import torch
import base64
from CNN import Net
from skimage import io
import seaborn as sns
import matplotlib.pyplot as plt
import base64

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

def getInput(imgData):
    convertImage(imgData)
    x = imread('output.png', mode='L')
    x = np.invert(x)
    x = imresize(x, (28, 28))
    x = x.astype('float32')
    x = x.reshape((1, 1, 28, 28))
    x = torch.from_numpy(x)
    return x

@app.route('/', methods=['GET', 'POST'])
def basic():

    return render_template('index.html')


@app.route('/capsule/', methods=['GET', 'POST'])
def predict():

    imgData = request.get_data()

    x = getInput(imgData)

    # perform the prediction
    caps_output = capsules_model(x)
    caps_output = caps_output.detach().numpy()
    caps = caps_output.flat
    
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.clf()
    sns.set_style('darkgrid')
    sns.barplot(x=values, y=list(caps), palette='Blues_d')
    plt.xlabel('Digits to be classified')
    plt.ylabel('Probabilities')
    plt.title('Capsule Networks')
    plt.savefig('./static/capsules.png')
    response =''
    with open('./static/capsules.png','rb') as file:
        response = base64.b64encode(file.read())
    return response

@app.route('/cnn/',methods=['GET','POST'])
def cnn_predict():
    imgData = request.get_data()

    x = getInput(imgData)
    cnn_output = cnn_model(x)
    cnn_output = cnn_output.detach().numpy()
    cnn_output = cnn_output *-1
    cnn = cnn_output.flat
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.clf()
    sns.set_style('darkgrid')
    sns.barplot(x=values, y=list(cnn), palette='Greens_d')
    plt.xlabel('Digits to be classified')
    plt.ylabel('Probabilities')
    plt.title('Convolutional Neural Networks')
    # cnn_graph.title("The probability of the lowest element represents the output")
    # cnn_graph.set_xlabel('Digits to be classified')
    # cnn_graph.set_ylabel('Probability')
    plt.savefig('./static/cnn.png')
    response =''
    with open('./static/cnn.png','rb') as file:
        response = base64.b64encode(file.read())
    return response

if __name__ == '__main__':
    app.run(threaded=True, debug=True, host='0.0.0.0')
