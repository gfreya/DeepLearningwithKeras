from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

base_model = VGG16(weights='imagenet',include_top=True)
for i,layer in enumerate(base_model.layers):
    print(i,layer.name,layer.output_shape)


model = Model(input=base_model.input,output=base_model.get_layer('block4_pool').output)
img_path='cat.jpg'
img=image.load_img(img_path,target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)

features = model.predict(x)