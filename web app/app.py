from flask import Flask, render_template,redirect,request
import keras
from keras.preprocessing import image
from keras.models import * 
import tensorflow as tf
import numpy as np

app = Flask(__name__)



@app.route('/')

def hello():
	return render_template("index.html")


@app.route('/submit', methods = ['POST'])
def final():
	if request.method == 'POST':
		f = request.files['userfile']
		f.save(f.filename)
        
		model = tf.keras.models.load_model('/home/vaibhav/covid detection/model/model_1')
		img = image.load_img( '/home/vaibhav/covid detection/web app/'+f.filename,target_size=(224,224))
		img = image.img_to_array(img)
		img = np.expand_dims(img,axis=0)
		answer = model.predict(img)
		pred = np.max(answer)
		if pred==0:
			ans = 'covid patient'
		else:
			ans = 'normal patient' 

	return render_template("output.html", detection_ans=ans, image_src ='/home/vaibhav/covid detection/web app/'+str(f.filename))		 	

if __name__ =='__main__':
	app.run(debug =True)

