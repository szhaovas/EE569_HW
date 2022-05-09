# ZHAO SHIHAN
# 5927678670
# shihanzh@usc.edu
# Apr 14

import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from LeNet5 import LeNet5
from matplotlib import pyplot as plt

# random_uniform, truncated_normal, zeros, glorot_normal
initializer = 'glorot_normal'
learning_rate = 0.002
decay = 0.9
batch_size = 128
epoch_num = 20

class LearningRate(Callback):
    def __init__(self, learning_rate, decay):
        super().__init__()
        self.learning_rate = learning_rate
        self.decay = decay

    def on_epoch_begin(self, epoch, logs={}):
        if epoch == 0:
            print("\nEpoch: {}. Setting learning rate {}".format(epoch, self.learning_rate))
            self.model.optimizer.lr.assign(self.learning_rate)

    def on_epoch_end(self, epoch, logs={}):
        old_lr = self.model.optimizer.lr.read_value()
        new_lr = old_lr * self.decay
        print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
        self.model.optimizer.lr.assign(new_lr)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    X_train = x_train.astype('float32')[:, :, :] / 255
    X_test = x_test.astype('float32')[:, :, :] / 255

    Y_train = to_categorical(y_train, 10)
    Y_test = to_categorical(y_test, 10)

    reload = input('Load existing model?(y/n)')
    if reload == 'y':
        model_filename = input("Model's filename?({blank} for default)")
        if model_filename == '':
            model = load_model('default_cifar')
        else:
            model = load_model(model_filename)

        cont = input("Continue training?(y/n)")
        cont = True if cont == 'y' else False
    else:
        model = LeNet5(X_train[0].shape, 10, initializer)
        cont = True

    if cont:
        callbacks_list = [
            LearningRate(learning_rate, decay),
            ModelCheckpoint('temp', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        ]

        history = model.fit(
            x=X_train,
            y=Y_train,
            batch_size=batch_size,
            epochs=epoch_num,
            validation_data=(X_test, Y_test),
            verbose=1,
            callbacks=callbacks_list
        )

        plt.figure()
        eps = [i for i in range(epoch_num)]
        plt.plot(eps, history.history['accuracy'], label='Training Accuracy')
        plt.plot(eps, history.history['val_accuracy'], label='Testing Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.savefig(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '1.png')
        model = load_model('temp')

model.summary()

classes = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]
predicted_labels = np.argmax(model(X_test), axis=1)
_, true_labels = np.where(Y_test == 1)
label_match = predicted_labels == true_labels
test_accuracy = np.sum(label_match) / X_test.shape[0]
confusion_matrix = np.zeros((10, 10))
for pred, truth in zip(predicted_labels, true_labels):
    # rows are actual labels; columns are predicted labels
    confusion_matrix[truth, pred] += 1

scaled_confusion_matrix = np.zeros((10, 10))
for r in range(10):
    scaled_confusion_matrix[r,:] = confusion_matrix[r,:] / np.sum(confusion_matrix, axis=1)[r]

with open(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '2.txt', 'w') as f:
    f.write(str(scaled_confusion_matrix)+'\n')

for d in range(10):
    scaled_confusion_matrix[d,d] = 0

fig, ax = plt.subplots()
heatmap = ax.pcolor(scaled_confusion_matrix, cmap=plt.cm.Blues)
for r in range(10):
    for c in range(10):
        if r == c:
            continue
        ax.text(
            c + 0.5, r + 0.5, int(confusion_matrix[r,c]),
            horizontalalignment='center',
            verticalalignment='center'
        )

ax.set_xlabel('Predicted Labels')
ax.xaxis.set_label_position('top')
ax.set_ylabel('Ground Truths')
fig.colorbar(heatmap)

ax.set_xticks(np.arange(10) + 0.5, minor=False)
ax.set_yticks(np.arange(10) + 0.5, minor=False)

ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xticklabels(classes, minor=False)
ax.set_yticklabels(classes, minor=False)
fig.savefig(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '2.png')

fig2, axs = plt.subplots(1,3)
for i in range(3):
    truth, pred = np.where(scaled_confusion_matrix == np.max(scaled_confusion_matrix))
    truth = truth[0]
    pred = pred[0]
    images = X_test[(predicted_labels != true_labels) & (true_labels == truth) & (predicted_labels == pred), :, :, :]
    axs[i].imshow(images[0,:,:,:].squeeze())
    axs[i].axis('off')
    axs[i].set_title('truth:{}\npred:{}'.format(classes[truth], classes[pred]))
    scaled_confusion_matrix[truth, pred] = 0
fig2.savefig(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '3.png')
