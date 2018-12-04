import matplotlib as mlp
mlp.use('Agg')

from keras.layers import Dense,Reshape,Flatten,Dropout,LeakyReLU,Input,Activation,BatchNormalization
from keras.models import Sequential,Model
from keras.layers.convolutional import Convolution2D,UpSampling2D
from keras.optimizers import Adam
from keras.regularizers import l1,l1_l2
from keras.datasets import mnist
import numpy as np
import pandas as pd
from keras import backend as k

from keras_adversarial import AdversarialModel,ImageGridCallback,simple_gan,gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous,normal_latent_sampling,AdversarialOptimizerAlternating
from image_utils import dim_ordering_fix,dim_ordering_input,dim_ordering_reshape,dim_ordering_unfix

def gan_target(n):
    """
    Stantard training targets [generator_fake,generator_real,discriminator_fake,discriminator_real]=[1,0,0,1]
    :param n: number of samples
    :return: array of targets
    """
    generator_fake = np.ones((n,1))
    generator_real = np.zeros((n,1))
    discriminator_fake = np.zeros((n,1))
    discriminator_real = np.ones((n,1))
    return [generator_fake,generator_real,discriminator_fake,discriminator_real]

def model_generator():
    nch = 256
    g_input = Input(shape=[100])
    H = Dense(nch*14*14,init='glorot_normal')(g_input)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = dim_ordering_reshape(nch,14)(H)
    H = UpSampling2D(size=(2,2))(H)
    H = Convolution2D(int(nch/2),3,3,border_mode='same',init='glorot_uniform')(H)
    H = BatchNormalization(mode=2,axis=1)(H)
    H = Activation('relu')(H)
    H = Convolution2D(int(nch/4),3,3,border_mode='same',init='glorot_uniform')(H)
    H = BatchNormalization(mode=2,axis=1)(H)
    H = Activation('relu')(H)
    H = Convolution2D(1,1,1,border_mode='same',init='glorot_uniform')(H)
    g_V = Activation('sigmoid')(H)
    return Model(g_input,g_V)

def model_discriminator(input_shape=(1,28,28),dropout_rate=0.5):
    d_input = dim_ordering_input(input_shape,name="input_x")
    nch = 512
    H = Convolution2D(int(nch/2),5,5,subsample=(2,2),border_mode='same',activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Convolution2D(nch,5,5,subsample=(2,2),border_mode='same',activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(1,activation='sigmoid')(H)
    return Model(d_input,d_V)

def mnist_process(x):
    x = x.astype(np.float32)/255.0
    return x

def mnist_data():
    (xtrain,ytrain),(xtest,ytest)=mnist.load_data()
    return mnist_process(xtrain),mnist_process(xtest)

if __name__ == "__main__":
    latent_dim = 100
    input_shape=(1,28,28)
    generator = model_generator()
    discriminator = model_discriminator(input_shape=input_shape)
    gan = simple_gan(generator,discriminator,normal_latent_sampling((latent_dim,)))
    generator.summary()
    discriminator.summary()
    gan.summary()

    model = AdversarialModel(base_model=gan,player_params=[generator.trainable_weights,discriminator.trainable_weights],player_names=["generator","discriminator"])
    model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),player_optimizers=[Adam(1e-4,decay=1e-4),Adam(1e-3,decay=1e-4)],loss='binary_crossentropy')

    def generator_sampler():
        zsamples = np.random.normal(size=(10*10,latent_dim))
        gen = dim_ordering_unfix(generator.predict(zsamples))
        return gen.reshape((10,10,28,28))

    generator_cb = ImageGridCallback("output/gan_convolutional/epoch-{:03d}.png",generator_sampler)
    xtrain,xtest=mnist_data()
    xtrain=dim_ordering_fix(xtrain.reshape((-1,1,28,28)))
    xtest=dim_ordering_fix(xtest.reshape((-1,1,28,28)))
    y = gan_targets(xtrain.shape[0])
    ytest = gan_targets(xtest.shape[0])
    history = model.fit(x=xtrain,y=y,validation_data=(xtest,ytest),callbacks=[generator_cb],nb_epoch=100,batch_size=32)
    df = pd.DataFrame(history.history)
    df.to_csv("output/gan_convolutional/history.csv")
    generator.save("output/gan_convolutional/generator.h5")
    discriminator.save("output/gan_convolutional/discriminator.h5")

    def dim_ordering_fix(x):
        if k.image_dim_ordering()=='th':
            return x
        else:
            return np.transpose(x,(0,2,3,1))