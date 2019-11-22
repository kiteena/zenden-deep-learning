from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, InputLayer
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD, Adadelta

def fitModel(model, train_X, train_y, epochs, batch_size): 
    model.fit(x=train_X, y=train_y, epochs=epochs, batch_size=batch_size)

def makeModelArchitectureOne(): 
    model = Sequential()
    model.add(Conv2D(input_shape=(64,64,3),filters=32, kernel_size=5, strides=1, padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=5, padding='same'))
    model.add(Conv2D(filters=50, kernel_size=5, strides=1, padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=5, padding='same'))
    model.add(Conv2D(filters=80, kernel_size=5, strides=1, padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=5, padding='same'))        
    model.add(Conv2D(filters=130, kernel_size=5, strides=1, padding='same',activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=5, padding='same'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, input_shape =(None,320), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(11, input_shape =(None,512),activation='softmax'))
    optimizer = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def makeModelArchitectureTwo(): 
    model = Sequential()
    model.add(Conv2D(input_shape=(64,64,3),filters=32, kernel_size=5, strides=1, padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=5, padding='same'))
    model.add(Conv2D(filters=50, kernel_size=5, strides=1, padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=5, padding='same'))
    model.add(Conv2D(filters=80, kernel_size=5, strides=1, padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=5, padding='same'))        
    model.add(Conv2D(filters=130, kernel_size=5, strides=1, padding='same',activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=5, padding='same'))    
    model.add(Conv2D(filters=210, kernel_size=5, strides=1, padding='same',activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=5, padding='same'))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, input_shape =(None,320), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(11, input_shape =(None,512),activation='softmax'))
    optimizer = SGD(lr=0.1)
#     optimizer = Adadelta()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model