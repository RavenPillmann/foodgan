import csv
import os

def copyImages():
	os.system("mkdir bucket/vegetable_images")

	with open('data.csv', 'r') as from_file:
		csv_reader = csv.reader(from_file)

		index = 0
		for row in csv_reader:
			_id = row[0]
			category = row[1]
			
			if category == 'vegetables':
				os.system("cp ./bucket/images/"+str(_id)+".jpg ./bucket/vegetable_images/"+str(index)+".jpg")
				index += 1


def main():
	copyImages()	


if __name__ == "__main__":
	main()