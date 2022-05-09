mnist.py generates all results concerning the mnist dataset, such as the accuracy plot, confusion matrix, and confused pairs. Simply python3 mnist.py and follow the queries given in the input prompt.

If the accuracy plot is needed, then the model must be retrained. Otherwise the already-trained default model can be used to generate confusion matrix etc.

Hyperparameters can be changed at the top of mnist.py. epoch_num represents the number of epochs to train, and try_num represents how many times to re-initialize the model and train from the start. The latter is only needed for parameter exploration and generating mean and standard deviation accuracies.

For adding noise to the training data, enter the noise probability as a decimal number, e.g. enter ε=40% as 0.4.

Since parameter exploration and noisy training data are not needed for the fashion-mnist and cifar10 datasets, fashion_mnist.py and cifar.py don't include these two functionalities, but the rest is pretty similar to mnist.py.