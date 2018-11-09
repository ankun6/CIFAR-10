from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from keras.models import Sequential
from keras import backend as K
from matplotlib import pylab as plt
import keras

batch_size = 32
epochs = 60
num_classes = 10

# 标签分类字典
label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}

# 加载数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 转换数据类型
x_train = x_train.astype('float32') / 255
x_test  = x_test.astype('float32') / 255

# 转换标签为One-hot
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)

# 搭建网络
mod = Sequential()
mod.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
mod.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
mod.add(MaxPooling2D(pool_size=(2, 2)))
mod.add(Dropout(0.25))
mod.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
mod.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
mod.add(MaxPooling2D(pool_size=(2, 2)))
mod.add(Dropout(0.25))
mod.add(Flatten())
mod.add(Dense(512, activation='relu'))
mod.add(Dropout(0.5))
mod.add(Dense(num_classes, activation='softmax'))
mod.summary()

# 优化器
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# 编译模型
mod.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# 训练
history = mod.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  shuffle=True)

# 绘制模型的精度和损失，可视化方便观察
fig = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('mod accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
fig.savefig('./visualization/mod_acc.png')
fig.clear()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('mod loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
fig.savefig('./visualization/mod_loss.png')
fig.clear()

# 训练结果评估
score = mod.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

# 打印评估结果
print('Test loss     : ', score[0])
print('Test accuracy : ', score[1])

# 保存模型和权重
mod.save('./mods/cifar10_mod.h5', overwrite=True)
mod.save_weights('./mods/cifar_mod_weights.h5', overwrite=True)

# 使用完模型之后，清空之前mod占用的内存
K.clear_session()
tf.reset_default_graph()

