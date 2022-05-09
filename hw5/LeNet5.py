# ZHAO SHIHAN
# 5927678670
# shihanzh@usc.edu
# Apr 14

import tensorflow.keras as keras

class LeNet5(keras.models.Sequential):
    def __init__(self, input_shape, num_categories, initializer):
        super().__init__()

        self.add(keras.layers.Conv2D(
            filters=6,
            kernel_size=5,
            activation='relu',
            kernel_initializer=initializer,
            input_shape=input_shape
        ))
        self.add(keras.layers.MaxPool2D(
            pool_size=(2,2)
        ))
        self.add(keras.layers.Conv2D(
            filters=16,
            kernel_size=5,
            activation='relu',
            kernel_initializer=initializer
        ))
        self.add(keras.layers.MaxPool2D(
            pool_size=(2,2)
        ))
        self.add(keras.layers.Flatten())
        self.add(keras.layers.Dense(
            units=120,
            activation='relu',
            kernel_initializer=initializer
        ))
        self.add(keras.layers.Dense(
            units=84,
            activation='relu',
            kernel_initializer=initializer
        ))
        self.add(keras.layers.Dense(
            units=num_categories,
            activation='softmax',
            kernel_initializer=initializer
        ))

        self.compile(
            optimizer='adam',
            loss=keras.losses.categorical_crossentropy,
            metrics=['accuracy']
        )
