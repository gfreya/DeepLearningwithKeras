from keras import backend as k
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD,RMSprop,Adam
import numpy as np
import matplotlib.pyplot as plt

# define ConvNet
class LeNet:
    @staticmethod
    def build(input_shape,classes):
        model = Sequential()
        # CONV=>RELU=>POOL
        model.add(Conv2D(20,kernel_size=5,padding="same",input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        # CONV=>RELU=>POOL
        model.add(Conv2D(50,kernel_size=5,border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        # Flatten=>RELU
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        # softmax
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

# Network and training
NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2
IMG_ROW,IMG_COLS = 28,28
NB_CLASSES = 10
INPUT_SHAPE = (1,IMG_ROW,IMG_COLS)

# divide training and testing set
(X_train,y_train),(X_test,y_test)=mnist.load_data()
k.set_image_dim_ordering("th")
# float and normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 225
X_test /= 225

X_train = X_train[:,np.newaxis,:,:]
X_test = X_test[:,np.newaxis,:,:]
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')

y_train = np_utils.to_categorical(y_train,NB_CLASSES)
y_test = np_utils.to_categorical(y_test,NB_CLASSES)

# initializing
model = LeNet.build(input_shape=INPUT_SHAPE,classes=NB_CLASSES)
model.compile(loss="categorical_crossentropy",optimizer=OPTIMIZER,metrics=["accuracy"])
history = model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=NB_EPOCH,verbose=VERBOSE,validation_split=VALIDATION_SPLIT)
score = model.evaluate(X_test,y_test,verbose=VERBOSE)
print("test score:",score[0])
print("test accuracy:",score[1])
# list all history data
print(history.history.keys())
# accuracy data
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

# loss data
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()