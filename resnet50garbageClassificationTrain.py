import keras
import matplotlib
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import glob
import matplotlib.pyplot as plt

# re-size all the images to this
IMAGE_SIZE = (224, 224, 3)

train_path = 'resnet50-Datasets/trainSet'
test_path = 'resnet50-Datasets/testSet'

# 添加预训练模型
resnet = ResNet50(input_shape=IMAGE_SIZE, weights=None, include_top=False)  # weights = 'imagenet'

# 固定预训练层的参数
# for layer in resnet.layers:
#     layer.trainable = False

# 通过文件夹获取类别信息
folders = glob.glob(train_path + '/*')
print(folders, len(folders))

# 添加自己的层
y = Flatten()(resnet.output)
prediction = Dense(len(folders), activation='softmax')(y)

# 创建自定义模型
model = Model(inputs=resnet.input, outputs=prediction)

# # 模型建立
# model = keras.models.Sequential()
# model.add(resnet)
# # # 添加自己的层
# x = Flatten()
# model.add(x)
# y = Dense(100, activation='relu')
# model.add(y)
# prediction = Dense(len(folders), activation='softmax')
# model.add(prediction)


# 查看模型的结构
print("------------------查看模型的结构------------------")
model.summary()

# 编译模型
model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

# 设置预处理数据规则
train_datagen = ImageDataGenerator(
    rescale=1. / 225,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1  # 训练集中分出0.1作为验证集
)
#
test_datagen = ImageDataGenerator(shear_range=0.1,
                                  zoom_range=0.1,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  )

# 产生训练集、验证集、测试集生成器
training_set = train_datagen.flow_from_directory(directory=train_path,
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 subset='training'
                                                 )
validation_set = train_datagen.flow_from_directory(directory=train_path,
                                                   target_size=(224, 224),
                                                   batch_size=32,
                                                   class_mode='categorical',
                                                   subset='validation'
                                                   )
test_set = test_datagen.flow_from_directory(directory=test_path,
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical',
                                            )


# 回调函数

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
        plt.savefig('./results/resnet50garbageClassificationTrain.png')


history = LossHistory()

# 得到训练集和验证集的图片数
training_set_nums = training_set.classes.size
validation_set_nums = validation_set.classes.size
print("训练集图片数{}".format(training_set_nums))
print("验证集图片数{}".format(validation_set_nums))
# 训练模型
model.fit(
    training_set,
    validation_data=validation_set,
    epochs=100,
    steps_per_epoch=training_set_nums // 32,
    validation_steps=validation_set_nums // 32,
    callbacks=[history]
)

history.loss_plot('epoch')
#
model.save('./results/resnet50garbageClassificationTrain.h5')
print('done')
