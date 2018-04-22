#!/usr/bin/env python
import numpy as np
import cv2
from matplotlib import pyplot as plt
from latest import cnn_training, image_to_digital
# import modules used here -- sys is a very standard one
import sys
import csv

refPts = np.zeros((1, 4), np.uint8)
image = np.zeros((512, 512, 3), np.uint8)
windowName = 'HW Window';
lx = -1
ly = -1
counter = 0
model = cnn_training()


def click_and_keep(event, x, y, flags, param):
	# grab references to the global variables
	global refPts, image,lx,ly, counter, model
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates 
	# performed
	if event == cv2.EVENT_LBUTTONDOWN and not(lx==x) and not(ly == y):
		refPts[0][counter] = x
		refPts[0][counter+1] = y
		lx = x
		ly = y
		counter = counter + 2
		print (refPts[0][0], refPts[0][2] ,refPts[0][3] ,refPts[0][1])
	
	if counter == 4:
		filename = turnIntoImage()
		classname = image_to_digital(filename, model)
		#at the end all the answers are in excel
		with open('output.csv', 'w') as outfile:
			writer = csv.writer(outfile)
			writer.writerow([classname])
		refPts = np.zeros((1,4), np.uint8)
		lx = -1
		ly = -1
		counter = 0

def turnIntoImage():
	global refPts, image
	y = refPts[0][0]
	h = refPts[0][2] - refPts[0][0]
	w = refPts[0][3] - refPts[0][1]
	x = refPts[0][1]
	print (y,h,w,x)
	crop_img = image[y:y+h, x:x+w]
	
	cv2.imwrite("single_unit.jpg", crop_img);
	return "single_unit.jpg"

# Gather our code in a main() function
def main():
	# Read Image
	image = cv2.imread('Dolly_W2.jpg',1);
	# image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	cv2.namedWindow(windowName)
	cv2.setMouseCallback(windowName, click_and_keep)
 
# keep looping until the 'c' key is pressed
	while True:
	# display the image and wait for a keypress
		image = cv2.circle(image,(lx,ly), 10, (0,255,255), -1);
		cv2.imshow(windowName, image)
		key = cv2.waitKey(1) & 0xFF
 
	# if the 'c' key is pressed, break from the loop
		if key == ord("c"):
			break
 	

	# Close the window will exit the program
	cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

