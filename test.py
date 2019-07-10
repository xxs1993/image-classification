import cv2
import os;

import keras
import numpy as np
from keras import Sequential
from keras.engine import InputLayer
from keras.layers import MaxPooling2D, Dropout, Flatten, Dense, Activation, Conv2D
from keras.optimizers import Adam


def train_data_set(path):
    data_set = []
    for dir in os.listdir(path):
        new_path = os.path.join(path,dir)
        for file in os.listdir(new_path):
            file_path = os.path.join(new_path,file)
            img = cv2.imread(file_path)
            img = cv2.resize(img,(224,224))
            data_set.append([np.array(img),dir])
    return data_set
train_data = train_data_set("E:/sperm_image/HuSHem/train")
test_data = train_data_set("E:/sperm_image/HuSHem/test")

train_data_img = np.array([i[0] for i in train_data]).reshape(-1,224,224,3).astype("float32")
train_data_label = np.array([i[1] for i in train_data]).reshape(-1,1)

test_data_img = np.array([i[0] for i in test_data]).reshape(-1,224,224,3).astype("float32")
test_data_label = np.array([i[1] for i in test_data]).reshape(-1,1)

train_data_label = keras.utils.to_categorical(train_data_label, num_classes=4)
test_data_label = keras.utils.to_categorical(test_data_label, num_classes=4)
vgg16_model = keras.applications.vgg16.VGG16(weights="D:/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
vgg16_model.summary()
model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)
# model.add(InputLayer(input_shape=[32,32,3]))
# model.add(Conv2D(32, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(4,activation="softmax"))
opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
model.compile(optimizer=opt,loss="categorical_crossentropy" ,metrics=["accuracy"])
# model.compile(optimizer=opt,loss="sparse_categorical_crossentropy" ,metrics=["accuracy"])

model.fit(train_data_img,train_data_label,epochs=30,batch_size=20)

model.save("E:/sperm_image/HuSHem/model.h5")

for i in test_data_img:
    i = np.expand_dims(i,0)
    result = model.predict(i)
    print(result)
print(test_data_label)