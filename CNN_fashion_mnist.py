import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
# 模型定义
model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation='relu', padding="same", input_shape=[28, 28, 1]),
    keras.layers.MaxPool2D(2),
    keras.layers.Conv2D(128, 3, activation='relu', padding="same"),
    keras.layers.Conv2D(128, 3, activation='relu', padding="same"),
    keras.layers.MaxPool2D(2),
    keras.layers.Conv2D(256, 3, activation='relu', padding="same"),
    keras.layers.Conv2D(256, 3, activation='relu', padding="same"),
    keras.layers.MaxPool2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax"),
])
model.summary()
# # 训练前的准备工作
# # 指定损失函数和优化器,训练评估指标（准确度）
# model.compile(loss=keras.losses.sparse_categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(),
#               metrics=[keras.metrics.sparse_categorical_accuracy],
#               )
#
# # 训练与评估
# history = model.fit()
