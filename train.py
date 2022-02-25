import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from keras.models import Sequential
from keras.models import load_model
import matplotlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob, os, random

base_path = './dataset-resized'
img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))
print(len(img_list))
batch_size_value = 32

train_datagen = ImageDataGenerator(
    rescale=1. / 225,  # rescale的作用是对图片的每个像素值均乘上这个放缩因子，这个操作在所有其它变换操作之前执行，
    # 在一些模型当中，直接输入原图的像素值可能会落入激活函数的“死亡区”，
    # 因此设置放缩因子为1/255，把像素值放缩到0和1之间有利于模型的收敛，避免神经元“死亡”。
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1)

test_datagen = ImageDataGenerator(
    rescale=1. / 255, validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
    base_path,
    target_size=(300, 300),
    batch_size=batch_size_value,
    class_mode='categorical',  # class_mode: "categorical", "binary", "sparse", "input" 或 None 之一。
    # 默认："categorical"。决定返回的标签数组的类型：
    subset='training',  # 数据子集 ("training" 或 "validation")，
    seed=0)

validation_generator = test_datagen.flow_from_directory(
    base_path,
    target_size=(300, 300),
    batch_size=batch_size_value,
    class_mode='categorical',
    subset='validation',
    seed=0)

labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
print(labels)

model = Sequential([
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(300, 300, 3)),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax')
])


class LossHistory(keras.callbacks.Callback):
    # 函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    # 按照batch来进行追加数据
    def on_batch_end(self, batch, logs={}):
        # 每一个batch完成后向容器里面追加loss，acc
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        # 每一个epoch完成后向容器里面追加loss，acc
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        # plt.show()
        plt.savefig('./results/learning_rate.png')


history = LossHistory()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit_generator(train_generator, epochs=100, steps_per_epoch=2276 // 32, validation_data=validation_generator,
                    validation_steps=251 // 32, callbacks=[history])
history.loss_plot('epoch')

model.save('./results/my_model1.h5')
