from scipy import misc
import numpy as np 
from glob import glob
import csv
import PIL
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

def load_image(file):
	
	img = PIL.Image.open(file)
	#img = misc.imread(file)	
	img = img.resize((100,100), PIL.Image.ANTIALIAS)
	#img = misc.imresize(img, (100,100))
	#img = np.reshape(img, (img_width, img_height, img_depth))
	img = np.array(img)
	img = img.astype(float)






	img = np.reshape(img, (100, 100, 3))


	img = img/255.0
	return img

def get_filenames(directory):
    return glob('{}/*.jpeg'.format(directory))


def get_label(labels, file):
	parts = file.split("/")
	
	fileName = parts[-1][:-5]
	label = labels[fileName]
	oneHot = [0]*5
	oneHot[int(label)] = 1
	return oneHot    

def get_labels():
    with open('trainLabels.csv', mode='r') as infile:
        reader = csv.reader(infile)
        labels = {rows[0]:rows[1] for rows in reader}
    return labels

def generator():
	filenames = get_filenames("stuff/www.kaggle.com/c/diabetic-retinopathy-detection/download/train")
	labels = get_labels()

	x_train = []
	y_train = []
	while 1:
		for (i,file) in enumerate(filenames):
			x_train.append(load_image(file))
			y_train.append(get_label(labels, file))

			if i % 5 == 0 and i != 0:
				yield np.array(x_train),np.array(y_train)
				x_train = []
				y_train = []

def get_model():
	model = Sequential()
	# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
	# this applies 32 convolution filters of size 3x3 each.
	model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(100, 100, 3)))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Convolution2D(64, 3, 3, border_mode='valid'))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	# Note: Keras does automatic shape inference.
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(5))
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

#
#generator = generator()

#train_x, train_y = next(generator)

model = get_model()
#model.fit_generator(generator(), batch_size=32, nb_epoch=10)
model.fit_generator(generator(), samples_per_epoch = 10, nb_epoch = 10, verbose=2, show_accuracy=True, callbacks=[], validation_data=None, class_weight=None, nb_worker=1)


