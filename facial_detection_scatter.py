# modules
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

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

# data storage
measures = open("emotions_data.txt", "w+")
columns = f'time_data,surpresa,neutro,raiva,feliz,triste,state\n'
measures.write(columns)

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

		# emotions
		surpresa = model.predict(np.array([face.reshape((48,48,1))]))[0][0]
		neutro = model.predict(np.array([face.reshape((48,48,1))]))[0][1]
		raiva = model.predict(np.array([face.reshape((48,48,1))]))[0][2]
		feliz = model.predict(np.array([face.reshape((48,48,1))]))[0][3]
		triste = model.predict(np.array([face.reshape((48,48,1))]))[0][4]

		# x axis
		if surpresa > raiva:
			x_ = surpresa
		else:
			x_ = -1 * raiva

		# y axis
		if feliz > triste:
			y_ = feliz
		else:
			y_ = -1 * triste
		
		# chart axis
		ax = plt.gca()
		ax.spines['left'].set_position('center')
		ax.spines['bottom'].set_position('center')
		ax.spines['right'].set_color('none')
		ax.spines['top'].set_color('none')
		ax.xaxis.set_ticks_position('bottom')
		ax.yaxis.set_ticks_position('left')
		ax.xaxis.set_major_locator(plt.MaxNLocator(4))
		ax.yaxis.set_major_locator(plt.MaxNLocator(4))
		
		ax.set_ylabel('Tristeza    -    Felicidade')
		ax.yaxis.set_label_coords(-0.05, 0.5)

		ax.set_xlabel('Raiva    -    Surpresa')
		ax.xaxis.set_label_coords(0.5, -0.05)
		
		plt.xlim(-1, 1)
		plt.ylim(-1, 1)

		# colors
		if x_ > 0 and y_ > 0:
			c_ = 'blue'
		elif x_ < 0 and y_ < 0:
			c_ = 'red'
		elif x_ > 0 and y_ < 0:
			c_ = 'yellow'
		else:
			c_ = 'green'
			
		# circle
		angle = np.linspace( 0 , 2 * np.pi , 150 )
		radius = 1
		x = radius * np.cos( angle )
		y = radius * np.sin( angle ) 

		# plot
		ax.plot(x, y, c = 'black', linestyle = 'dashed', linewidth=0.9)
		ax.scatter(x_, y_, color = c_)
		
		# if i % 10 == 0:
		# 	plt.clf()
		
		# storage
		time_data = datetime.now()
		emotions_data = f'{time_data},{surpresa},{neutro},{raiva},{feliz},{triste},{state}\n'
		measures.write(emotions_data)
	
	## show plots and imags
	plt.savefig('face_scatterplot.png')
	img_face_scatterplot = cv2.imread('face_scatterplot.png')
	
	# show face classification
	cv2.imshow('Facial Expression', img)
	cv2.imshow('Scatter plot probabilities', img_face_scatterplot)
	

	if cv2.waitKey(5) != -1:
		measures.close()
		break

camera.release()
cv2.destroyAllWindows()
