from keras.models import load_model
from keras.datasets import cifar10
from matplotlib import pylab as plt
import keras
import tensorflow as tf

index = 2345

# 分类标签字典
label_dict = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
              5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

# 加载数据
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 加载模型
mod = load_model('./mods/cifar10_mod.h5')

data = x_test[index:index + 1]

# 预测分类
result = mod.predict(data, 1, 0)
category = mod.predict_classes(data, 1, 1)

# 输出结果
print('P: ', result.max())
print('Category: ', category, ' ', label_dict.get(int(category)))
print('Y: ', y_test[index])
plt.imshow(data.reshape(32, 32, 3))
plt.show()

# 清理
keras.backend.clear_session()
tf.reset_default_graph()
