import os
import cv2
import csv
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D

epochs = 1
# Steering offset for views from left side and right side.
# This appeared to be crucial in the model training.
correction_dict = {
    0 : 0,
    1: 0.4,
    2: -0.4
}

samples = []
with open('./data/driving_log4.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#Get sizes of training image_# height, width, number of channels in image
print(len(samples))
token1 =line[0].split(';')
image_path1 =token1[0]
img = cv2.imread(image_path1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_height   = img.shape[0]
img_width    = img.shape[1]
img_channels = img.shape[2]
print("Original Image Sizes=",img_height,img_width,img_channels)        
ch, row, col = img_channels, img_height, img_width
        
train_samples, validation_samples = train_test_split(samples, test_size = 0.2) 

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]

            images, angles = [], []
            for batch_sample in batch_samples:
                token=batch_sample[0].split(';')
                for i in range(3):
                    file_path = token[i]
                    image = cv2.imread(file_path)
                    b,g,r = cv2.split(image)
                    image = cv2.merge([r,g,b])
                    if image is not None:
                        angle = float(token[3]) + correction_dict[i]
                        images.append(image)
                        angles.append(angle)
                        images.append(cv2.flip(image, 1))
                        angles.append(angle * -1.0)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)
            print(len(X_train))
train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)


print("Model is starting...")
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch = len(train_samples) * 6, validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = epochs)
model.save('model.h5')

