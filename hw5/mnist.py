# ZHAO SHIHAN
# 5927678670
# shihanzh@usc.edu
# Apr 14

import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from LeNet5 import LeNet5
from matplotlib import pyplot as plt

# random_uniform, truncated_normal, zeros, glorot_normal
initializer = 'truncated_normal'
learning_rate = 0.005
decay = 0.9
batch_size = 64
epoch_num = 5
try_num = 5

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
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    X_train = x_train.astype('float32')[:, :, :, np.newaxis] / 255
    X_test = x_test.astype('float32')[:, :, :, np.newaxis] / 255

    Y_train = to_categorical(y_train, 10)
    Y_test = to_categorical(y_test, 10)

    reload = input('Load existing model?(y/n)')
    if reload == 'y':
        model_filename = input("Model's filename?({blank} for default)")
        if model_filename == '':
            model = load_model('default_mnist')
        else:
            model = load_model(model_filename)
    else:
        noisy = input('Add noise to training set?({ε}/n)')
        if noisy != 'n':
            one_hot10 = np.diag(np.ones(10))
            _, train_labels = np.where(Y_train == 1)

            noise = float(noisy)
            for i in range(Y_train.shape[0]):
                probabilities = [noise/9 for _ in range(10)]
                probabilities[train_labels[i]] = 1 - noise
                new_label = np.random.choice(10, p = probabilities)
                Y_train[i] = one_hot10[new_label, :]

        callbacks_list = [
            LearningRate(learning_rate, decay),
            ModelCheckpoint('temp', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        ]

        history_arr = np.zeros((try_num,epoch_num))
        test_history_arr = np.zeros((try_num,epoch_num))
        for i in range(try_num):
            model = LeNet5(X_train[0].shape, 10, initializer)
            history = model.fit(
                x=X_train,
                y=Y_train,
                batch_size=batch_size,
                epochs=epoch_num,
                validation_data=(X_test, Y_test),
                verbose=1,
                callbacks=callbacks_list
            )
            history_arr[i,:] = history.history['accuracy']
            test_history_arr[i,:] = history.history['val_accuracy']

        means = np.mean(history_arr, axis=0)
        stds = np.std(history_arr, axis=0)
        test_means = np.mean(test_history_arr, axis=0)
        test_stds = np.std(test_history_arr, axis=0)
        plt.figure()
        eps = [i for i in range(epoch_num)]
        plt.errorbar(eps, means, yerr=stds, label='Training Accuracy')
        plt.errorbar(eps, test_means, yerr=test_stds, label='Testing Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.savefig(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '1.png')
        model = load_model('temp')

        with open(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '1.txt', 'w') as f:
            f.write('BEST:\n')
            f.write(str(np.max(history_arr))+'\n')
            f.write('MEAN:\n')
            f.write(str(means)+'\n')
            f.write('STANDARD DEVIATION:\n')
            f.write(str(stds)+'\n')

model.summary()

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

# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
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

ax.set_xticklabels([i for i in range(10)], minor=False)
ax.set_yticklabels([i for i in range(10)], minor=False)
fig.savefig(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '2.png')

fig2, axs = plt.subplots(1,3)
for i in range(3):
    truth, pred = np.where(scaled_confusion_matrix == np.max(scaled_confusion_matrix))
    truth = truth[0]
    pred = pred[0]
    images = X_test[(predicted_labels != true_labels) & (true_labels == truth) & (predicted_labels == pred), :, :, :]
    axs[i].imshow(images[0,:,:,:].squeeze())
    axs[i].axis('off')
    axs[i].set_title('truth:{} pred:{}'.format(truth, pred))
    scaled_confusion_matrix[truth, pred] = 0
fig2.savefig(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '3.png')
