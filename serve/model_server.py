# Following https://medium.com/dataswati-garage/deploy-your-machine-learning-model-as-api-in-5-minutes-with-docker-and-flask-8aa747b1263b

import os
import json
from flask import Flask, jsonify, request
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import argparse
from google.cloud import storage
import time
import cv2

HEADERS = {'Content-type': 'application/json', 'Accept': 'text/plain'}
# model = None

argument_parser = argparse.ArgumentParser(description="Run Server")
argument_parser.add_argument("--model", help='Model File (H5 format)', dest='model')
args = vars(argument_parser.parse_args())


def flask_app(model):
	app = Flask(__name__)

	@app.route('/', methods=['GET'])
	def server_up():
		return "server up"


	def generate_noise(n_samples, noise_dim):
		X = np.random.normal(0, 1, size=(n_samples, noise_dim))
		return X


	def store_image(img):
		# TODO: Put image into bucket
		# Create the image url
		# Pass back
		filename = str(int(time.time())) + ".jpg"
		# TODO: Save image
		cv2.imwrite(filename, cv2.cvtColor(np.around(img).astype('uint8'), cv2.COLOR_BGR2RGB))
		# im = Image.fromarray(img)
		# im.save("./static/"+filename)


		os.system("gsutil cp " + filename + " gs://recipe_images_111/" + filename)

		return "https://storage.cloud.google.com/recipe_images_111"+filename


	@app.route('/get_image', methods=['POST'])
	def get_img():
		# req = request.json

		# TODO: Get ingredients from request, check matches
		# Alternatively, randomly choose ingredients

		# Create GAN, save it to url
		noise = generate_noise(1, 100)
		input_class_section = np.ones((1, 19))
		ingreds = np.random.randint(0, 2, [1, 50])
		classes = np.random.randint(0, 19, 1)
		input_class_section[0, classes[0]] = 0
    
		input_class_section = np.hstack((input_class_section, ingreds))
  
		noise = np.hstack((noise, input_class_section))
  		
		with graph.as_default():
			gen_img = model.predict(noise)
			img = image.array_to_img(gen_img[0])
			url = store_image(img)

			return jsonify({"image": url})

	return app


def main():
#	os.system("mkdir static")
	model = load_model(args['model'])
	global graph
	graph = tf.get_default_graph()
	app = flask_app(model)
	app.run(debug=True, host='0.0.0.0', port=80)


if __name__ == '__main__':
	main()
