from sklearn.datasets import load_sample_image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# load sample images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])
# 4d tensor
batch_size, height, width, channels = images.shape

# create 2 filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # vertical
filters[3, :, :, 1] = 1  # horizontal
outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")
plt.imshow(outputs[0, :, :, 1], cmap="gray")  # plot 1st image's 2nd feature map
plt.show()
plt.imshow(outputs[0, :, :, 0], cmap="gray")
plt.show()
plt.imshow(outputs[1, :, :, 1], cmap="gray")
plt.show()
plt.imshow(outputs[1, :, :, 0], cmap="gray")
plt.show()
