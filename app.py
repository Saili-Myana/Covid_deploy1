#Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import cv2
import h5py
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os


# Create flask instance
app = Flask(__name__)

#Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init():
    global graph
    graph = tf.compat.v1.get_default_graph()
    print('hello0')



# Function to load and prepare the image in right shape
def read_image(file_path):

	print('reading image')

	img = cv2.imread(file_path)
	print('hello1')
	img = cv2.resize(img, (300, 300))
	print('hello2')
    # Load the image
	#img = load_img(filename, target_size=(300, 300))
    # Convert the image to array
	img = np.asarray(img)
	print('hello3')
    # Reshape the image into a sample of 1 channel
	img = img.reshape(1, 300, 300, 3)
	print(img.shape)
    # Prepare it as pixel data
	#img = img / 255.0
	return img

 


@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])


def predict():
    if request.method == 'POST':
        file = request.files['file']
        print('0')
        try:
            if file and allowed_file(file.filename):
            	print('1'+'\n')
            	filename = file.filename
            	file_path = os.path.join('images', filename)
            	file.save(file_path)
            	print('2'+'\n')
            	print(file_path)
            	print(type(file_path))
            	print(filename)
            	img = read_image(file_path)
            	print('3'+'\n')
                # Predict the class of an image

            	with graph.as_default():
            		print('4'+'\n')

            		model = load_model('COVID_MODEL_RELU.h5')
            		#with open('COVID_MODEL_ReLU (1).pkl', 'rb') as f:
            		#	model = pickle.load(f)
						
            		print('model loaded')
            		print('5'+'\n')
            		prediction = model.predict(img)
            		print('6'+'\n')
            		print(prediction)

                #Map apparel category with the numerical class
            	if (prediction[0][0] >= 0.5):
                	product = "Normal"
            	else:
                	product = "Covid affected"
            	return render_template('predict.html', product = product, user_image = file_path)
        except Exception as e:
            return "Unable to read the file. Please check if the file extension is correct."

    return render_template('predict.html')

if __name__ == "__main__":
    init()
    app.run()