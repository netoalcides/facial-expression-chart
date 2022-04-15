# modules
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# classifier type
face_detection = cv2.CascadeClassifier('haar_cascade_face_detection.xml')

# access camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
settings = {
	'scaleFactor': 1.3, 
	'minNeighbors': 5, 
	'minSize': (50, 50)
}

# model
#labels = ['Surprise', 'Neutral', 'Anger', 'Happy', 'Sad']
labels = ['Surpresa', 'Neutro', 'Raiva', 'Feliz', 'Triste']
model = tf.keras.models.load_model('network-5Labels.h5')

# detection
while True:

	# camera
	ret, img = camera.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	detected = face_detection.detectMultiScale(gray, **settings)

	# face rectangule
	for x, y, w, h in detected:
		cv2.rectangle(img, (x, y), (x+w, y+h), (245, 135, 66), 2)
		cv2.rectangle(img, (x, y), (x+w//3, y+20), (245, 135, 66), -1)
		face = gray[y+5:y+h-5, x+20:x+w-20]
		face = cv2.resize(face, (48,48)) 
		face = face/255.0
		
		# model prediction
		predictions = model.predict(np.array([face.reshape((48,48,1))])).argmax()
		state = labels[predictions]

		# face classification
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,state,(x+10,y+15), font, 0.5, (255,255,255), 2, cv2.LINE_AA)

		## chart creation

		# barplot
		probs = model.predict(np.array([face.reshape((48,48,1))]))[0]
		plt.bar(labels, probs)

		# # test
		# surpresa = model.predict(np.array([face.reshape((48,48,1))]))[0][0]
		# neutro = model.predict(np.array([face.reshape((48,48,1))]))[0][1]
		# raiva = model.predict(np.array([face.reshape((48,48,1))]))[0][2]
		# feliz = model.predict(np.array([face.reshape((48,48,1))]))[0][3]
		# triste = model.predict(np.array([face.reshape((48,48,1))]))[0][4]

		# x_ = surpresa - raiva
		# y_ = feliz - triste
		
		# ax_test.scatter(x_, y_)
		
		# if i % 10 == 0:
		# 	plt.clf()
	
	## show plots and imags
	plt.savefig('face_barplot.png')
	plt.close()
	img_face_barplot = cv2.imread('face_barplot.png')
	
	
	# show face classification
	cv2.imshow('Facial Expression', img)
	cv2.imshow('Bar plot probabilities', img_face_barplot)
	

	if cv2.waitKey(5) != -1:
		break

camera.release()
cv2.destroyAllWindows()
