from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import os

# 创建日志目录
root_logdir = os.path.join(os.curdir, "my_logs")


# 定义日志创建函数
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()

# 数据导入和预处理
housing = fetch_california_housing()
x_train_full, x_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
x_train, x_valid, y_train, y_valid = train_test_split(x_train_full, y_train_full)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test = scaler.transform(x_test)

# 使用顺序API构建模型
# 构建网络
model = keras.models.Sequential(
    [
        keras.layers.Dense(30, activation="relu", input_shape=x_train.shape[1:]),
        keras.layers.Dense(1),
    ]
)

# 指定损失函数、优化器、训练评估指标（准确度）
model.compile(optimizer=keras.optimizers.SGD(),
              loss=keras.losses.mean_squared_error,
              metrics=keras.metrics.mean_squared_error,
              )

# 训练模型
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid), callbacks=[tensorboard_cb])

# 评估模型
mes_test = model.evaluate(x_test, y_test)

# 使用模型预测

x_new = x_test[:3]
y_pred = model.predict(x_new)
print(y_pred)

#可视化
keras.utils.plot_model(model, to_file='net.png', show_shapes=True)
