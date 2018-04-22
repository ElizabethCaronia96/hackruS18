import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Dropout, Conv2D, GRU, Activation, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

def cnn_training():
	(train_x, train_y) , (test_x, test_y) = mnist.load_data()
	train_x = train_x.reshape(60000,784)
	test_x = test_x.reshape(10000,784)
	train_y = keras.utils.to_categorical(train_y,10)
	test_y = keras.utils.to_categorical(test_y,10)
	model = Sequential()
	model.add(Dense(units=128,activation="relu",input_shape=(784,)))
	model.add(Dense(units=128,activation="relu"))
	model.add(Dense(units=128,activation="relu"))
	model.add(Dense(units=10,activation="softmax"))
	#model.add(Dense(units=10,activation="softmax"))
	model.compile(optimizer=SGD(0.001),loss="categorical_crossentropy",metrics=["accuracy"])
	model.load_weights("mnistmodel.h5")
	#model.fit(train_x,train_y,batch_size=32,epochs=10,verbose=1)
	#model.save("mnistmodel.h5")
	return model

def classy_model(model):
	# build a classifier model to put on top of the convolutional model
	(train_x, train_y) , (test_x, test_y) = mnist.load_data()
	train_x = train_x.reshape(60000,784)
	test_x = test_x.reshape(10000,784)
	train_y = keras.utils.to_categorical(train_y,10)
	test_y = keras.utils.to_categorical(test_y,10)
	top_model = Sequential()
	top_model.add(Dense(128, activation='relu', input_shape = (784,)))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(units=10,activation="softmax"))
	top_model.compile(optimizer=SGD(0.001),loss="categorical_crossentropy",metrics=["accuracy"])

	top_model.fit(train_x,train_y,batch_size=32,epochs=10,verbose=1)
	top_model.save("topmodel.h5")

	# note that it is necessary to start with a fully-trained
	# classifier, including the top classifier,
	# in order to successfully do fine-tuning
	#top_model.load_weights("topmodel.h5")

	# add the model on top of the convolutional base
	model.add(top_model)
	return model
# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
def update(model):
	for layer in model.layers[:25]:
	    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
	model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


	model.fit(train_x,train_y,batch_size=32,epochs=10,verbose=1)

def image_to_digital(path, model):
	img = tf.keras.preprocessing.image.load_img(path,grayscale=True,target_size=(28,28,1))
	img = tf.keras.preprocessing.image.img_to_array(img)
	print (img.shape)
	test_img = img.reshape((1,784))
	img_class = model.predict_classes(test_img)
	prediction = img_class[0]
	classname = img_class[0]
	print("Class: ",classname)
	return classname
if __name__ == '__main__':
	
	model = cnn_training()
	#model = classy_model(model)
	#update(model)