from keras_preprocessing import image
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

"""
这个类是做什么用的？通过实时数据增强生成张量图像数据批次，并且可以循环迭代，我们知道在Keras中，
当数据量很多的时候我们需要使用model.fit_generator()方法，该方法接受的第一个参数就是一个生成器。
简单来说就是：ImageDataGenerator()是keras.preprocessing.image模块中的图片生成器，
可以每一次给模型“喂”一个batch_size大小的样本数据，同时也可以在每一个批次中对这batch_size个样本数据进行增强，
扩充数据集大小，增强模型的泛化能力。比如进行旋转，变形，归一化等等。
总结起来就是两个点：
（1）图片生成器，负责生成一个批次一个批次的图片，以生成器的形式给模型训练；
（2）对每一个批次的训练图片，适时地进行数据增强处理（data augmentation）；
"""
base_path = "dataset-resized"
# img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))
# img_list = glob.glob("dataset-resized/*/*.jpg")
# print(img_list)
# 构建ImageDataGenerator类的对象
train_datagen = image.ImageDataGenerator(rescale=1. / 255,  #
                                         shear_range=0.1,
                                         zoom_range=0.1,
                                         width_shift_range=0.1,
                                         height_shift_range=0.1,
                                         horizontal_flip=True,
                                         vertical_flip=True, )
# 第二步：调用flow_from_directory方法产生处理后数据

train_generator = train_datagen.flow_from_directory(directory=base_path,  # 图片目录
                                                    target_size=(300, 300),  # 将图片大小转换
                                                    batch_size=100,  # 一批数据的大小（默认 32）
                                                    class_mode="categorical",  # 对类别进行独热编码
                                                    save_to_dir="testData/1",  # 将augmentation之后的图片保存位置
                                                    save_prefix="aug_",
                                                    )
# 输出标签
labels = (train_generator.class_indices)
print(labels)
labels = dict((v, k) for k, v in labels.items())
print(labels)
# 第三步：迭代训练集的图片数据，进行augmentation,迭代查看
# count = 1
# for x_batch, y_batch in train_generator:
#     print(F"------------开始第{count}次迭代-----------------------------")
#     print(F"------------x_batch、y_batch的形状如下----------------------")
#     print(np.shape(x_batch), np.shape(y_batch))
#     print('-------------y_batch打印结果如下-----------------------------')
#     print(y_batch)
#     print('============================================================')
#
#     # 将每次augmentation之后的5幅图像显示出来
#     for i in range(5):
#         plt.subplot(1, 5, i + 1)
#         plt.imshow(x_batch[i].reshape(300, 300, 3))
#         plt.savefig("testData\\{}.png".format(count), format="png")
#     plt.show()
#     count += 1
#     if count > 100:
#         break
