from PIL import Image
import argparse
import numpy as np
import time
from tensorflow.keras.models import load_model
import cv2

parser = argparse.ArgumentParser(description="Test the model")

parser.add_argument("--model_file", "-m", dest="model_file", help="The saved file of the model")
parser.add_argument("--dest", "-d", dest="dest", help="Destination directory for image")

args = parser.parse_args()
args = vars(args)

model_file = args['model_file']
dest = args['dest']

model = load_model(model_file)

# food_array = np.zeros((1, 20))
# food_array = np.random.randint(0, 2, size=[1, 20])
# food_array[0, 0] = 1

noise = np.random.uniform(-1., 1., size=[1, 50])

# input_array = np.concatenate((noise, food_array), axis=1)
input_array = noise

image_output = model.predict(input_array)
image_output = np.clip(image_output[0] * 255., 0, 255)

print(image_output.shape)


output_file_name = dest + "/" + str(int(time.time()))+".jpg"

cv2.imshow("generated_image", image_output.astype(np.uint8))
cv2.waitKey(5000)
cv2.imwrite(output_file_name, image_output)

# img = Image.fromarray(image_output.astype(np.uint8))
# b, g, r = img.split()
# img = Image.merge("RGB", (r, g, b))
# img.save(dest + "/" +output_file_name)
# img.show()