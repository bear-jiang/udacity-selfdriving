import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import Reshape

def data_genarator(data, batch_size):
    batch_size = batch_size//2
    data_length = len(data)
    while True:
        data = shuffle(data)
        for offset in range(0,data_length,batch_size):
            data_batch = data[offset:offset+batch_size,:]
            images = []
            angles = []
            for line in data_batch:
#                 image_name = 'data/' + line[0]
                image_name  ='/input/IMG/' + line[0].split('/')[-1]
                image = mpimg.imread(image_name)
                images.append(image)
                images.append(image[:,::-1,:])
                
                angles.append(1.2*line[-1])
                angles.append(-1.2*line[-1])
                
            yield shuffle(np.array(images), np.array(angles))

csv_data = pd.read_csv('/input/driving_log.csv')
train_data, valid_data = train_test_split(csv_data.values[:,(0,3)], test_size=0.2)

train_generator = data_genarator(train_data, batch_size=64)
validation_generator = data_genarator(valid_data, batch_size=64)

model = Sequential()
model.add(Lambda(lambda x: x[:,:,:,0]*0.299+x[:,:,:,1]*0.587+x[:,:,:,2]*0.114, input_shape=(160,320,3)))
model.add(Reshape((160, 320, 1)))
model.add(Lambda(lambda x: (x-128)/128, input_shape=(160,320,1)))
model.add(Convolution2D(20, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(40, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(60, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(120, 3, 3, activation='relu'))
model.add(Convolution2D(120, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch= 2*len(train_data), validation_data=validation_generator, nb_val_samples=2*len(valid_data), nb_epoch=5)
model.save('model.h5')