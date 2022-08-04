#Name: Samar Aswani
#Task: Facial Recognition using OpenCV


import sys
import cv2
import os
import numpy as np

subjects = ["", "Samar Aswani", "Rishi Barve", "Mahesh Aswani"]
def detect_face(img):
#convert the test image to gray scale as opencv face detector expects gray images
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
#load OpenCV face detector (LBP)
	#face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 
#This code detects multiscale images(some images may be closer to camera than others)
#result is a list of faces
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
 
#if no faces are detected then return original img
	if (len(faces) == 0):
		return None, None
 
#under the assumption that there will be only one face,
#extract the face area
	(x, y, w, h) = faces[0]
 
#return only the face part of the image
	return gray[y:y+w, x:x+h], faces[0]


data_folder_path = "C:/samaraswani/documents/FacialRecognitionProb/training-data/"

def prepare_training_data(data_folder_path):
 

#get the directories (one directory for each subject) in data folder
	dirs = os.listdir(data_folder_path)
	print(dirs)
 
#list to hold all subject faces
	faces = []
#list to hold labels for all subjects
	labels = []
 
#let's go through each directory and read images within it
	for dir_name in dirs:
 
#our subject directories start with letter 's' so
#ignore any non-relevant directories if any
		if not dir_name.startswith("s"):
			continue;
 
		label = int(dir_name.replace("s", ""))
 
#build path of directory containing images for current subjects
#sample subject_dir_path = "training-data/s1"
		subject_dir_path = data_folder_path + "/" + dir_name
 
#get the images names that are inside the given subject directory
		subject_images_names = os.listdir(subject_dir_path)
 	
#go through each image name, read image, 
#detect face and add face to list of faces
		for image_name in subject_images_names:
 
#ignore system files like .DS_Store
			if image_name.startswith("."):
				continue;
 
#build image path
#sample image path = training-data/s1/1.pgm
			image_path = subject_dir_path + "/" + image_name
 
#read image
			image = cv2.imread(image_path)
 
#display an image window to show the image 
			cv2.imshow("Training on image...", image)
			cv2.waitKey(100)
 
#detect face
			face, rect = detect_face(image)
 
#ignores faces that are not detected
			if face is not None:
#add face to list of faces
				faces.append(face)
#add label for this face
				labels.append(label)
 
	cv2.destroyAllWindows()
	cv2.waitKey(1)
	cv2.destroyAllWindows()
 
	return faces, labels	

print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")
 
#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create() 
face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img):
#make a copy of the image as we don't want to change original image
	img = test_img.copy()
#detect face from the image
	face, rect = detect_face(img)

#predict the image using the coded face recognizer above
	label= face_recognizer.predict(face)
#get name of respective label returned by face recognizer
	label_text = subjects[label[0]]
	
#draw a rectangle around face detected
	draw_rectangle(img, rect)
#draw name of predicted person
	draw_text(img, label_text, rect[0], rect[1]-5)

 
	return img




print("Predicting images...")
 


#load test images. 
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
test_img3 = cv2.imread("test-data/test3.jpg")
#perform a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
print("Prediction complete")
 
#display both images
cv2.imshow(subjects[1], predicted_img1)
cv2.imshow(subjects[2], predicted_img2)
cv2.imshow(subjects[3], predicted_img3)
cv2.waitKey(0)
sys.exit()

	
