import argparse
import skimage as sk
import random
import os
import numpy as np

parser = argparse.ArgumentParser(description="Image Augmenter")

parser.add_argument("--input_directory", "-i", dest="input", help="Input image directory")
parser.add_argument("--output_directoy", "-o", dest="output", help="Output image directory")

args = parser.parse_args()
args = vars(args)

input_dir = args['input']
output_dir = args['output']

# Following https://medium.com/@thimblot/data-augmentation-boost-your-image-dataset-with-few-lines-of-python-155c2dc1baec

def random_rotation(image):
	degree = random.uniform(-25, 25)
	return sk.transform.rotate(image, degree)


def horizontal_flip(image):
	return image[:, ::-1]


def generate_images():
	filepaths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, filename))]

	index = 0
	for filepath in filepaths:
		image = sk.io.imread(filepath)

		rotation_one = random_rotation(image)
		rotation_two = random_rotation(image)
		flip = horizontal_flip(image)
		rotation_three = random_rotation(flip)
		rotation_four = random_rotation(flip)

		write_to_filepath_one = os.path.join(output_dir, str(index) + ".jpg")
		index += 1
		write_to_filepath_two = os.path.join(output_dir, str(index) + ".jpg")
		index += 1
		write_to_filepath_three = os.path.join(output_dir, str(index) + ".jpg")
		index += 1
		write_to_filepath_four = os.path.join(output_dir, str(index) + ".jpg")
		index += 1
		write_to_filepath_five = os.path.join(output_dir, str(index) + ".jpg")
		index += 1
		write_to_filepath_six = os.path.join(output_dir, str(index) + ".jpg")
		index += 1

		sk.io.imsave(write_to_filepath_one, image.astype(np.uint8))
		sk.io.imsave(write_to_filepath_two, rotation_one)
		sk.io.imsave(write_to_filepath_three, rotation_two)
		sk.io.imsave(write_to_filepath_four, flip.astype(np.uint8))
		sk.io.imsave(write_to_filepath_five, rotation_three)
		sk.io.imsave(write_to_filepath_six, rotation_four)


if __name__ == "__main__":
	generate_images()