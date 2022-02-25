# import tensorflow as tf
# import numpy as np
# from tensorflow import keras
# from keras_preprocessing import image
# import os

# imgdir = "dataset-resized\cardboard"

# 通过keras_preprocessing.image的图像预处理


# # 此方法会将一个文件下面的所有图片名称返回，以一个列表的形式，
# # 第一个参数为目录名称，第二个为图片的相关拓展名，
# img_list = image.list_pictures(imgdir)
# print(img_list)


# 加载图像
# img01 = image.load_img("dataset-resized/cardboard/cardboard1.jpg")
# # keras.preprocessing.image.ImageDataGenerator()
# print(img01.format, img01.size)
# # 显示图像
# img01.show()


# # 保存图像
# # 自己定义的一个（4,4,3）的numpy数组
# img_num = np.array([[[10, 30, 60], [100, 120, 150], [77, 99, 130], [200, 30, 59]],
#                     [[40, 10, 160], [150, 120, 150], [77, 99, 130], [100, 30, 59]],
#                     [[20, 90, 210], [100, 220, 150], [37, 199, 230], [210, 90, 99]],
#                     [[100, 40, 40], [200, 50, 20], [157, 9, 140], [50, 230, 119]]])
#
# image.save_img("testData/1.jpg", img_num, scale=True)  # scale默认为true
# image.save_img("testData/2.jpg", img_num, scale=False)  # scale设置为false


# 将PIL.Image图像对象和numpy数组互转
# img02 = image.load_img("dataset-resized/cardboard/cardboard100.jpg")
# img02_np = image.img_to_array(img02)
# print(img02_np)
# #
# img03 = image.array_to_img(img02_np)
# img03.show()


# 对图片进行仿射变换
# 1、随机旋转random_rotation()
# 2、随机平移random_shift()
# 3、随机错切random_shear()
# 4、随机缩放random_zoom()
# 5、根据变换矩阵进行仿射变换transform_matrix_offset_center(matrix, x, y)方法


# 其他预处理方法
# apply_channel_shift(x, intensity, channel_axis=0):
# random_channel_shift(x, intensity_range, channel_axis=0)
# apply_brightness_shift(x, brightness):
# random_brightness(x, brightness_range):
# flip_axis(x, axis):
# from keras_preprocessing import image
# import glob
# import os
# base_path = "dataset-resized"
# img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))
# img_list = glob.glob("dataset-resized/*/*.jpg")
# print(len(img_list))
# import tensorflow as tf
#
# print(tf.__version__)
# # 查看是否支持GPU
# import tensorflow as tf
# import sys
# print(sys.version)
# print(tf.__version__)
# print(tf.config.list_physical_devices('GPU'))
# print(tf.test.is_built_with_cuda())
# # import tensorflow as tf
# # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

####################################################################
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import glob
import keras
import matplotlib.pyplot as plt

train_path = 'resnet50-Datasets/trainSet'
test_path = 'resnet50-Datasets/testSet'

# add preprocessing layer to the front of resnet
resnet = ResNet50(input_shape=(300, 300, 3), weights='imagenet', include_top=False)

# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False

# 通过文件夹获取类别信息
folders = glob.glob(test_path + '/*')  # *为通配符
print(folders, len(folders))

# 模型建立
model = keras.models.Sequential()
model.add(resnet)

# # 添加自己的层
x = Flatten()
model.add(x)
y = Dense(100, activation='relu')
model.add(y)
prediction = Dense(len(folders), activation='softmax')
model.add(prediction)

# 查看模型的结构
print("------------------查看模型的结构------------------")
model.summary()
# # 编译模型
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )
#
# # 预处理数据，产生数据的可迭代的生成器
# train_datagen = ImageDataGenerator(rescale=1. / 255,
#                                    shear_range=0.1,
#                                    zoom_range=0.1,
#                                    horizontal_flip=True)
#
# test_datagen = ImageDataGenerator(rescale=1. / 255)
#
# training_set = train_datagen.flow_from_directory(directory=train_path,
#                                                  target_size=(300, 300),
#                                                  batch_size=32,
#                                                  class_mode='categorical')
#
# test_set = test_datagen.flow_from_directory(directory=test_path,
#                                             target_size=(300, 300),
#                                             batch_size=32,
#                                             class_mode='categorical')
#
#
# # 回调函数
#
# # class LossHistory(keras.callbacks.Callback):
# #     # 函数开始时创建盛放loss与acc的容器
# #     def on_train_begin(self, logs={}):
# #         self.losses = {'batch': [], 'epoch': []}
# #         self.accuracy = {'batch': [], 'epoch': []}
# #         self.val_loss = {'batch': [], 'epoch': []}
# #         self.val_acc = {'batch': [], 'epoch': []}
# #
# #     # 按照batch来进行追加数据
# #     def on_batch_end(self, batch, logs={}):
# #         # 每一个batch完成后向容器里面追加loss，acc
# #         self.losses['batch'].append(logs.get('loss'))
# #         self.accuracy['batch'].append(logs.get('acc'))
# #         self.val_loss['batch'].append(logs.get('val_loss'))
# #         self.val_acc['batch'].append(logs.get('val_acc'))
# #
# #     def on_epoch_end(self, batch, logs={}):
# #         # 每一个epoch完成后向容器里面追加loss，acc
# #         self.losses['epoch'].append(logs.get('loss'))
# #         self.accuracy['epoch'].append(logs.get('acc'))
# #         self.val_loss['epoch'].append(logs.get('val_loss'))
# #         self.val_acc['epoch'].append(logs.get('val_acc'))
# #
# #     def loss_plot(self, loss_type):
# #         iters = range(len(self.losses[loss_type]))
# #         plt.figure()
# #         # acc
# #         plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
# #         # loss
# #         plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
# #         if loss_type == 'epoch':
# #             # val_acc
# #             plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
# #             # val_loss
# #             plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
# #         plt.grid(True)
# #         plt.xlabel(loss_type)
# #         plt.ylabel('acc-loss')
# #         plt.legend(loc="upper right")
# #         # plt.show()
# #         plt.savefig('./results/resnet50garbageClassificationTrain.png')
# #
# #
# # history = LossHistory()
#
# # 得到训练集和验证集的图片数
# training_set_nums = training_set.classes.size
# test_set_nums = test_set.classes.size
# # 训练模型
# model.fit(
#     training_set,
#     validation_data=test_set,
#     epochs=50,
#     steps_per_epoch=training_set_nums//32,
#     validation_steps=test_set_nums//32,
#     #callbacks=[history]
# )
# #
# model.save('./results/resnet50garbageClassificationTrain.h5')
