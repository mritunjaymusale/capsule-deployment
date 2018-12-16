from flask import Flask, render_template, request, jsonify
from scipy.misc import imsave, imread, imresize
import numpy as np
import re
from CapsNet import capsules
import torch

device = torch.device("cpu")
cuda=False

A, B, C, D = 64, 8, 16, 16

model = capsules(A=A, B=B, C=C, D=D, E=10,
                     iters=2,cuda=False).to(device)
model.load_state_dict(torch.load('./saved_model/mnist.pth'))    
model.eval()                 
app = Flask(__name__)


def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', imgData1).group(1)
    # print(imgstr)
    with open('output.png', 'wb') as output:
        output.write(imgstr.decode('base64'))

@app.route('/', methods=['GET', 'POST'])
def basic():
    
    return render_template('index.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    #whenever the predict method is called, we're going
    #to input the user drawn character as an image into the model
    #perform inference, and return the classification
    #get the raw data format of the image
    imgData = request.get_data()
    #encode it into a suitable format
    
    convertImage(imgData)
    print("debug")
    #read the image into memory
    x = imread('output.png', mode='L')
    #compute a bit-wise inversion so black becomes white and vice versa
    x = np.invert(x)
    #make it the right size
    x = imresize(x, (28, 28))
    #imshow(x)
    #convert to a 4D tensor to feed into our model
    x = x.reshape(1, 28, 28, 1)
    print("debug2")
    #in our computation graph

    #perform the prediction
    out = model.predict(x)
    print(out)
    print(np.argmax(out, axis=1))
    #print "debug3"
    #convert the response to a string
    response = np.array_str(np.argmax(out, axis=1))
    return response

if __name__ == '__main__':
    app.run(threaded=True,debug=True)
