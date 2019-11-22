from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, InputLayer
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import to_categorical
from keras.optimizers import Adam, SGD, Adadelta
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Dot, Add, Flatten, Concatenate
from keras.layers import Dropout, BatchNormalization, Activation
from keras.regularizers import l2, l1, l1_l2
from keras.optimizers import SGD, Adam

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

def recommenderModel(K, num_users, num_houses, users, houses, matches):
    u = Input(shape=(1,))
    m = Input(shape=(1,))
    u_embedding = Embedding(num_users[0], K, embeddings_regularizer=l1(0))(u) 
    m_embedding = Embedding(num_houses[0], K, embeddings_regularizer=l1(0))(m)
    u_bias = Embedding(num_users[0], 1, embeddings_regularizer=l1(0))(u) 
    m_bias = Embedding(num_houses[0], 1, embeddings_regularizer=l1(0))(m) 
    x = Dot(axes=2)([u_embedding, m_embedding])
    x = Add()([x, u_bias, m_bias])
    x = Flatten()(x) # (N, 1)
    x = Dropout(0.5)(x)
    model = Model(inputs=[u, m], outputs=x)
    model.compile(
      loss='mse',
    #   optimizer='adam',
    #     optimizer=Adam(lr=0.1),
        optimizer=SGD(lr=0.9, momentum=0.8),
      metrics=['accuracy'],
    )

    r = model.fit(
      x=[users, houses],
      y=matches,
      epochs=100,
      batch_size=64,
      validation_data=(
        [users[500:1000], houses[500:1000]], 
          matches[500:1000]
      )
    )
    return model