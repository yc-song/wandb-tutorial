# if h5py is not installed, run "pip install h5py" first.
import h5py
import numpy as np

def load_data(one_hot_encoding=True):
    # load MNIST data
    MNIST_data = h5py.File("mnist/MNISTdata.hdf5", 'r')
    X_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:, 0])).reshape(-1, 1)
    X_test = np.float32(MNIST_data['x_test'][:])
    y_test = np.int32(np.array(MNIST_data['y_test'][:, 0])).reshape(-1, 1)
    MNIST_data.close()

    num_train = y_train.shape[0]
    num_test = y_test.shape[0]

    if one_hot_encoding:
        # stack train and test for one-hot encoding
        #X = np.vstack((x_train, x_test))
        y = np.vstack((y_train, y_test))

        # one-hot encoding
        num_classes = 10
        num_examples = num_train + num_test
        y = y.reshape(1, num_examples)
        y_new = np.eye(num_classes)[y.astype('int32')]
        y_new = y_new.T.reshape(num_classes, num_examples)

        # split again to train/test sets
        #X_train, X_test = X[:num_train], X[num_train:]
        y_train, y_test = y_new[:, :num_train].T, y_new[:, num_train:].T

    # shuffle the training set
    #shuffle_index = np.random.permutation(num_train)
    #X_train, Y_train = X_train[shuffle_index, :], Y_train[shuffle_index, :]

    print('MNIST data loaded:')

    print('Training data shape: {}'.format(X_train.shape))
    print('Training labels shape: {}'.format(y_train.shape))
    print('Test data shape: {}'.format(X_test.shape))
    print('Test labels shape: {}'.format(y_test.shape))

    return X_train, y_train, X_test, y_test

