# 1D 2D 3D CNN for hyperspectral image classification
# author: Tong Li,Ziye Wang
# contact: Ziye Wang (Email: ziyewang@cug.edu.cn)

import numpy as np
import scipy.io as scio
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, \
    Conv3D, MaxPooling3D
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# parameters setting
data_form = 1 # chosen of 1D CNN (1), 2D CNN (2), or 3D CNN (3)
window_size = 21 # the window size of training samples
num_classes = 8 # number of category
learning_rate = 1e-3 # initial learning rate
optimizer = Adam(learning_rate=learning_rate) # optimizer
pre_batch = 5000

# input data
def sample_generator(data_format):
    rawdata = scio.loadmat('./image.mat') # input hyperspectral image
    rawdata = rawdata['image']
    label = scio.loadmat('./class.mat') # input ground truth image
    label = label['class']

    "train"
    train_sample = []
    val_sample = []
    train_label = []
    val_label = []

# Training sample preparation
    for i in range(1, 9):
        loc = np.argwhere(label == i)
        if data_format == 1:
            rand = np.random.randint(0, len(loc), int(len(loc) * 0.1)) # Proportion of training samples
            randloc = loc[rand]
            for idx, lo in enumerate(randloc):
                if idx <= int(len(randloc) * 0.8):
                    train_sample.append(rawdata[lo[0], lo[1], :])
                    train_label.append([i])
                else:
                    val_sample.append(rawdata[lo[0], lo[1], :])
                    val_label.append([i])
        else:
            rand = np.random.randint(0, len(loc), int(len(loc) * 0.1)) # Proportion of training samples
            randloc = loc[rand]
            np.random.shuffle(randloc)
            for idx, lo in enumerate(randloc):
                data = rawdata[lo[0] - int(window_size / 2):lo[0] + int(np.ceil(window_size / 2)),
                       lo[1] - int(window_size / 2):lo[1] + int(np.ceil(window_size / 2)), :]
                if len(data.flatten()) == window_size ** 2 * rawdata.shape[2]:
                    if idx <= int(len(randloc) * 0.8):
                        train_sample.append(data)
                        train_label.append([i])
                    else:
                        val_sample.append(data)
                        val_label.append([i])
    train_sample = np.array(train_sample, dtype='float32')
    val_sample = np.array(val_sample, dtype='float32')
    if data_format == 3:
        train_sample = train_sample.reshape((-1, window_size, window_size, rawdata.shape[2], 1))
        val_sample = val_sample.reshape((-1, window_size, window_size, rawdata.shape[2], 1))
    return train_sample, val_sample, np.asarray(train_label) - 1, np.asarray(val_label) - 1

# CNN architecture
def cnn_1d():
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(30, 1))) # 30 is the dimension of hyperspectral image
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def cnn_2d():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(window_size, window_size, 30)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def cnn_3d():
    model = Sequential()
    model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same',
                     input_shape=(window_size, window_size, 30, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# model training
def predict(model):
    rawdata = scio.loadmat('./image.mat') # input remote sensing data to be classified
    rawdata = rawdata['image']
    result = []
    if data_form == 1:
        data = rawdata.reshape((rawdata.shape[0] * rawdata.shape[1], -1))
        for i in tqdm(range(0, len(data), pre_batch)):
            batch = data[i:i + pre_batch]
            batch_predictions = model.predict(batch, verbose=0)
            result.extend(batch_predictions)
        result = np.argmax(result, axis=1)
        result = np.array(result).reshape((rawdata.shape[0], rawdata.shape[1]))
        plt.imshow(result, cmap='jet')
        plt.show()
    else:
        for i in range(rawdata.shape[0] - window_size + 1):
            data = []
            for j in range(rawdata.shape[1] - window_size + 1):
                data.append(rawdata[i:i + window_size, j:j + window_size, :])
            data = np.array(data)
            if data_form == 3:
                data = data.reshape((-1, window_size, window_size, rawdata.shape[2], 1))
            batch_predictions = model.predict(data, verbose=0)
            result.extend(batch_predictions)
        result = np.argmax(result, axis=1)
        result = np.array(result).reshape((rawdata.shape[0]-window_size+1, rawdata.shape[1]-window_size+1)) # output result
        plt.imshow(result, cmap='jet')
        plt.show()
    return result

# model prediction
if __name__ == '__main__':
    data_t, data_v, label_t, label_v = sample_generator(data_format=data_form)
    if data_form == 1:
        model = cnn_1d()
    elif data_form == 2:
        model = cnn_2d()
    else:
        model = cnn_3d()
        history = model.fit(data_t, label_t, validation_data=(data_v, label_v), epochs=300, batch_size=128) # define the epoch and batch size

# plot accuracy and loss curves
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    result = predict(model)

    plt.plot(np.squeeze(train_loss), label='Training set')
    plt.plot(np.squeeze(val_loss), label='Validation set')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.show()

    plt.plot(np.squeeze(train_acc), label='Train set')
    plt.plot(np.squeeze(val_acc), label='Validation set')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.show()

# save result
    np.savetxt('./train_loss.csv', np.squeeze(train_loss))
    np.savetxt('./val_loss.csv', np.squeeze(val_loss))
    np.savetxt('./train_acc.csv', np.squeeze(train_acc))
    np.savetxt('./val_acc.csv', np.squeeze(val_acc))
    np.savetxt('./result.csv', result)