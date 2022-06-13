import numpy as np
from sklearn.model_selection import train_test_split
from imutils import paths
from keras.preprocessing.image import img_to_array,array_to_img
from keras.utils import np_utils, plot_model
from pylab import rcParams
import cv2
import os
from keras.models import *  
from keras.layers import *
from keras.preprocessing.image import img_to_array,array_to_img
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras
import math

# path of Data_gray_select
word_path = "..\\Data_gray_select"
word_list = os.listdir(word_path)
labels = []
data = []

count = 0
for word_num in range(0, len(word_list)):
    image_loads = sorted(list(paths.list_images(word_path + "\\" + word_list[word_num] + "\\")))
    for image_num in range(0, len(image_loads)):
        image = cv2.imdecode(np.fromfile(image_loads[image_num], dtype=np.uint8), -1)
        #image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
        image = img_to_array(image)
        data.append(image)
#         data[count] = image
        labels.append(word_num)
        count += 1

(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.1, stratify= labels)



datagen = ImageDataGenerator(
    zca_whitening = False,
    rotation_range = 90,
# #     width_shift_range=0.2,
# #     height_shift_range=0.2,
#     shear_range=1.5,
#     zoom_range=0.2,
    horizontal_flip = False,
    fill_mode='nearest')

img_test = []
labl = []
for idx in range(0, len(x_train)):
    img = x_train[idx]
    # img = array_to_img(x_train[0])
    img = img.reshape((1,) + img.shape)
    i = 0
    for batch in datagen.flow(img, batch_size=10):
        img_test.append(img)
        labl.append(y_train[idx])
        i += 1
        if i >= 6:
            break

for idx in range(0, len(img_test)):
    tmp = img_test[idx]
    img_test[idx] = tmp[0,:,:,:]

x_train.extend(img_test)
y_train.extend(labl)

H_Big = 90  
W_Big = 175 
for idx in range(0, len(x_train)):  
    image = x_train[idx]
    image = image[:,:,0]
    if image.shape[0] < H_Big:
        padding = np.ones(( (H_Big - image.shape[0]), image.shape[1]))*255
        image = np.vstack((image, padding))
    if image.shape[1] < W_Big:
        fill_left = math.ceil((W_Big - image.shape[1]) / 2) 
        fill_right = (W_Big - image.shape[1]) - fill_left

        fill_left_ones = np.ones(( image.shape[0], fill_left))*255
        fill_right_ones = np.ones(( image.shape[0], fill_right))*255

        image = np.hstack((fill_left_ones, image))
        image = np.hstack((image, fill_right_ones))
    image = img_to_array(image)
    x_train[idx] = image

for idx in range(0, len(x_test)):  
    image = x_test[idx]
    image = image[:,:,0]
    if image.shape[0] < H_Big:
        padding = np.ones(( (H_Big - image.shape[0]), image.shape[1]))*255
        image = np.vstack((image, padding))
    if image.shape[1] < W_Big:
        fill_left = math.ceil((W_Big - image.shape[1]) / 2) 
        fill_right = (W_Big - image.shape[1]) - fill_left

        fill_left_ones = np.ones(( image.shape[0], fill_left))*255
        fill_right_ones = np.ones(( image.shape[0], fill_right))*255

        image = np.hstack((fill_left_ones, image))
        image = np.hstack((image, fill_right_ones))
    image = img_to_array(image)
    x_train[idx] = image


x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)

### model 2 vgg16

inputs = Input(shape=(90, 175, 1))  # 1
mg_conc = Concatenate()([inputs, inputs, inputs]) 
vgg16_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_tensor=mg_conc)
fc = Flatten()(vgg16_model.layers[16].output)
fc = Dense(512, activation='relu')(fc)
fc = Dense(512, activation='relu')(fc)
output = Dense(class_num, activation='softmax')(fc)
res_model = Model(inputs, output)
res_model.trainable = True

res_model.summary() 


from keras.optimizers import Adam, SGD
opt = Adam(lr=0.0001)

res_model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
train_history = res_model.fit(x = x_train, 
                              y = y_train_onehot,
                              validation_data=(x_test, y_test_onehot),
                              epochs=50, 
                              batch_size=150, 
                              verbose=1)


res_model.save('model_aug.h5')

