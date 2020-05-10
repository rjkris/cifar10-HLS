# !/usr/bin/env python
# coding:utf-8
# Author: Lizechen
# cifar10小模型

import tensorflow as tf
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, \
    LeakyReLU
from tensorflow.keras import Model
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 完全显示参数，确保提取的参数没有省略号
np.set_printoptions(threshold=np.inf)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = tf.convert_to_tensor(x_train)
x_test = tf.convert_to_tensor(x_test)
y_train = tf.squeeze(y_train, axis=1)
y_test = tf.squeeze(y_test, axis=1)

image_gen_train = ImageDataGenerator(
    rescale=1,  # 归至0～1
    rotation_range=0,  # 随机0度旋转
    width_shift_range=0,  # 宽度偏移
    height_shift_range=0,  # 高度偏移
    horizontal_flip=True,  # 水平翻转
    zoom_range=1  # 将图像随机缩放到100％
)
image_gen_train.fit(x_train)

class MiniModel(Model):
    def __init__(self):
        super(MiniModel, self).__init__()
        # 32,32,3
        self.c1 = Conv2D(filters=32, kernel_size=(5, 5), padding="valid", input_shape=(32, 32, 3))
        self.a1 = Activation("relu")
        # 28,28,32
        self.c2 = Conv2D(filters=32, kernel_size=(5, 5), padding="valid")
        self.a2 = Activation("relu")
        # 24,24,32
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")

        # 12,12,32
        self.c3 = Conv2D(filters=32, kernel_size=(3, 3), padding="valid")
        self.a3 = Activation("relu")
        # 10,10,32
        self.c4 = Conv2D(filters=32, kernel_size=(3, 3), padding="valid")
        self.a4 = Activation("relu")
        # 8,8,32
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        # 4,4,32
        self.flatten = Flatten()
        # 512
        self.f1 = Dense(256, activation='relu')
        self.f2 = Dense(128, activation='relu')
        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.a2(x)
        x = self.p1(x)
        x = self.c3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.a4(x)
        x = self.p2(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)

        return y


model = MiniModel()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./checkpoint/mnist.ckpt"

# 断点续训
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 monitor='val_loss',
                                                 save_best_only=True,
                                                 verbose=2)
history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=1,
                    validation_data=(x_test, y_test),
                    validation_freq=1, callbacks=[cp_callback], verbose=1)

model.summary()  # 打印网络结构
file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    # file.write(str(v.numpy()) + '\n')
    # 将参数扩大1024倍，存为一维数组用逗号隔开
    file.write(",".join(str(i) for i in (v.numpy().flatten() * 1024).astype(int).tolist()) + '\n')
file.close()

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
