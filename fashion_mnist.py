import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from tensorflow import keras

# 导入数据
fashion_mnist = keras.datasets.fashion_mnist
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train_full.shape,y_train_full.shape)
# 划分训练集为训练集和验证集,像素强度映射到0-1
x_valid, x_train = x_train_full[:5000] / 255.0, x_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
# 建立分类名列表
class_name = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
# 建立神经网络
# 创建一个Sequential模型，这是用于神经网络的最简单的Keras模型，
# 它仅由顺序连接的单层堆栈组成，这称为顺序API。
model = keras.models.Sequential()
# 我们构建第一层并将其添加到模型中。它是Flatten层，
# 其作用是将每个输入图像转换为一维度组，如果接收到输入数据X,则计算X.reshape(-1,1)。该层没有任何参数。
# 它只是在那里做一些简单的预处理。由于它是模型的第一层，因此应指定input_shape,其中不包括批处理大
# 小，而仅包括实例的形状。或者，你可以添加keras.layers.InputLayer作为第一层，设置input_shape=[28,28]。
model.add(keras.layers.Flatten(input_shape=[28, 28]))
# 中间Dense隐藏层设置，激活函数为ReLU
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
# Dense输出层使用softmax激活函数（排他多分类）
model.add(keras.layers.Dense(10, activation='softmax'))
"""
通过列表的方式可以一次设置所有层
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
])
"""
# 查看模型
model.summary()
# hidden1=model.layers[1]
# print(hidden1.get_weights())


# # 训练前的准备工作
# # 指定损失函数和优化器,训练评估指标（准确度）
# model.compile(loss=keras.losses.sparse_categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(),
#               metrics=[keras.metrics.sparse_categorical_accuracy],
#               )
# # 训练与评估
# history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))
# # print(history.epoch,history.history)
# # 可视化结果
# pd.DataFrame(history.history).plot(figsize=(8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# plt.show()
# # 对模型进行评估，在测试集上看其泛化误差
# print('------------------对模型进行评估，在测试集上看其泛化误差-------------------')
# model.evaluate(x_test, y_test)
# # 使用模型进行预测
# print('------------------使用模型进行预测-------------------')
# x_new = x_test[:3]
# y_proba = model.predict(x_new)
# print(y_proba.round(2))
# # 可视化
# keras.utils.plot_model(model, to_file='net.png', show_shapes=True)
