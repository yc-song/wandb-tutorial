# conda env create -f env.yml
# conda activate wandb-tutorial

import copy
import numpy as np
from mnist.data_utils import load_data
import os
import wandb
import json
import argparse
import time

# Utils
def sigmoid(z):
    """
    Do NOT modify this function
    """
    return 1 / (1 + np.exp(-z))


def softmax(X):
    """
    Do NOT modify this function
    """
    logit = np.exp(X - np.amax(X, axis=1, keepdims=True))
    numer = logit
    denom = np.sum(logit, axis=1, keepdims=True)
    return numer / denom


def load_batch(X, Y, batch_size, shuffle=True):
    """
    Generates batches with the remainder dropped.
    Do NOT modify this function
    """
    if shuffle:
        permutation = np.random.permutation(X.shape[0])
        X = X[permutation, :]
        Y = Y[permutation, :]
    num_steps = int(X.shape[0]) // batch_size
    step = 0
    while step < num_steps:
        X_batch = X[batch_size * step:batch_size * (step + 1)]
        Y_batch = Y[batch_size * step:batch_size * (step + 1)]
        step += 1
        yield X_batch, Y_batch


# 2-Layer Network
class TwoLayerNN:
    """ a neural network with 2 layers """

    def __init__(self, input_dim, num_hiddens, num_classes):
        """
        Do NOT modify this function.
        """
        self.input_dim = input_dim
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        self.params = self.initialize_parameters(input_dim, num_hiddens, num_classes)

    def initialize_parameters(self, input_dim, num_hiddens, num_classes):
        """
        initializes parameters with Xavier Initialization.
        Question (a)
        - refer to https://paperswithcode.com/method/xavier-initialization for Xavier initialization 
        
        Inputs
        - input_dim
        - num_hiddens
        - num_classes
        Returns
        - params: a dictionary with the initialized parameters.
        """
        params = {"W1": np.random.uniform(-1 / np.sqrt(input_dim), 1 / np.sqrt(input_dim), (input_dim, num_hiddens)),
                  "W2": np.random.uniform(-1 / np.sqrt(num_hiddens), 1 / np.sqrt(num_hiddens),
                                          (num_hiddens, num_classes)), "b1": np.zeros(num_hiddens),
                  "b2": np.zeros(num_classes)}

        return params

    def forward(self, X):
        """
        Define and perform the feed forward step of a two-layer neural network.
        Specifically, the network structue is given by
          y = softmax(sigmoid(X W1 + b1) W2 + b2)
        where X is the input matrix of shape (N, D), y is the class distribution matrix
        of shape (N, C), N is the number of examples (either the entire dataset or
        a mini-batch), D is the feature dimensionality, and C is the number of classes.
        Question (b)
        - ff_dict will be used to run backpropagation in backward method.
        Inputs
        - X: the input matrix of shape (N, D)
        Returns
        - y: the output of the model
        - ff_dict: a dictionary with all the fully connected units and activations.
        """
        ff_dict = {}
        ff_dict = self.params
        ff_dict["h"] = sigmoid((np.matmul(X, ff_dict["W1"]) + ff_dict["b1"]))

        ff_dict["minus h"] = 1 - sigmoid((np.matmul(X, ff_dict["W1"]) + ff_dict["b1"]))

        y = softmax(np.matmul(ff_dict["h"], ff_dict["W2"]) + ff_dict["b2"])
        ff_dict["y"] = y

        return y, ff_dict

    def backward(self, X, Y, ff_dict):
        """
        Performs backpropagation over the two-layer neural network, and returns
        a dictionary of gradients of all model parameters.
        Question (c)
        Inputs:
         - X: the input matrix of shape (B, D), where B is the number of examples
              in a mini-batch, D is the feature dimensionality.
         - Y: the matrix of one-hot encoded ground truth classes of shape (B, C),
              where B is the number of examples in a mini-batch, C is the number
              of classes.
         - ff_dict: the dictionary containing all the fully connected units and
              activations.
        Returns:
         - grads: a dictionary containing the gradients of corresponding weights and biases.
        """

        grads = {}
        grads["h"] = np.matmul(ff_dict["y"] - Y, np.transpose(ff_dict["W2"]))
        grads["dW2"] = np.matmul((np.transpose(ff_dict["h"])), ff_dict["y"] - Y)
        grads["db1"] = grads["h"] * ff_dict["h"] * ff_dict["minus h"]
        grads["db1"] = grads["db1"].sum(0)
        grads["dW1"] = np.matmul(np.transpose(X), grads["h"] * ff_dict["h"] * ff_dict["minus h"])
        grads["db2"] = (ff_dict["y"] - Y)
        grads["db2"] = grads["db2"].sum(0)

        return grads

    def compute_loss(self, Y, Y_hat):
        """
        Computes cross entropy loss.
        Do NOT modify this function.
        Inputs
            Y:
            Y_hat:
        Returns
            loss:
        """
        loss = -(1 / Y.shape[0]) * np.sum(np.multiply(Y, np.log(Y_hat)))
        return loss

    def train(self, X, Y, X_val, Y_val, lr, n_epochs, batch_size, log_interval=1):
        """
        Runs mini-batch gradient descent.
        Do NOT Modify this method.
        Inputs
        - X
        - Y
        - X_val
        - Y_Val
        - lr
        - n_epochs
        - batch_size
        - log_interval
        """

        for X_batch, Y_batch in load_batch(X, Y, batch_size):
            self.train_step(X_batch, Y_batch, batch_size, lr)
            Y_hat, ff_dict = self.forward(X)
            train_loss = self.compute_loss(Y, Y_hat)
            train_acc = self.evaluate(Y, Y_hat)
            Y_hat, ff_dict = self.forward(X_val)
            valid_loss = self.compute_loss(Y_val, Y_hat)
            valid_acc = self.evaluate(Y_val, Y_hat)

            print('train loss/acc: {:.3f} {:.3f}, valid loss/acc: {:.3f} {:.3f}'. \
                  format(train_loss, train_acc, valid_loss, valid_acc))
            return train_loss, train_acc, valid_loss, valid_acc

    def train_step(self, X_batch, Y_batch, batch_size, lr):
        """
        Updates the parameters using gradient descent.
        Do NOT Modify this method.
        Inputs
        - X_batch
        - Y_batch
        - batch_size
        - lr
        """
        _, ff_dict = self.forward(X_batch)
        grads = self.backward(X_batch, Y_batch, ff_dict)
        self.params["W1"] -= lr * grads["dW1"] / batch_size
        self.params["b1"] -= lr * grads["db1"] / batch_size
        self.params["W2"] -= lr * grads["dW2"] / batch_size
        self.params["b2"] -= lr * grads["db2"] / batch_size

    def evaluate(self, Y, Y_hat):
        """
        Computes classification accuracy.
        
        Do NOT modify this function
        Inputs
        - Y: A numpy array of shape (N, C) containing the softmax outputs,
             where C is the number of classes.
        - Y_hat: A numpy array of shape (N, C) containing the one-hot encoded labels,
             where C is the number of classes.
        Returns
            accuracy: the classification accuracy in float
        """
        classes_pred = np.argmax(Y_hat, axis=1)
        classes_gt = np.argmax(Y, axis=1)
        accuracy = float(np.sum(classes_pred == classes_gt)) / Y.shape[0]
        return accuracy


# Load MNIST
def main(args):
    # load json config file
    config = args
#     if config["resume"]:
#         run = wandb.init(project="wandb-tutorial", config=config, resume="must", id=config["run_id"])
#     else:
#         run = wandb.init(project="wandb-tutorial", config=config, notes="Hello, wandb!", tags=["tutorial"])
#         model_output_path = "./training_params/"+run.id
#         if not os.path.exists(model_output_path):
#             os.makedirs(model_output_path)
#         with open(os.path.join(model_output_path, "training_params.json"), 'w') as outfile:
#             json.dump(config, outfile)

#     print(run.id)
    print("config:", config)
    start_time = time.time()
    time_limit = 0.1 # minutes
    # load and shuffle data
    x_train, y_train, x_test, y_test = load_data()
    idxs = np.arange(len(x_train))
    np.random.shuffle(idxs)
    split_idx = int(np.ceil(len(idxs) * 0.8))
    x_valid, y_valid = x_train[idxs[split_idx:]], y_train[idxs[split_idx:]]
    x_train, y_train = x_train[idxs[:split_idx]], y_train[idxs[:split_idx]]
    print('Set validation data aside')
    print('Training data shape: ', x_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Validation data shape: ', x_valid.shape)
    print('Validation labels shape: ', y_valid.shape)

    # model instantiation
    model = TwoLayerNN(input_dim=784, num_hiddens=256, num_classes=10)

    # train the model"
    lr, n_epochs, batch_size = config["lr"], config["n_epochs"], config["batch_size"]
    # lr, n_epochs, batch_size = wandb.config.lr, wandb.config.n_epochs, wandb.config.batch_size
    print("check hyper parameters")
    print("learning rate: ", lr)
    print("n_epochs: ", n_epochs)
    print("batch size: ", batch_size)

    for epoch in range(n_epochs):
#         execution_time = (time.time() - start_time) / 60
#         print("execution time:", execution_time)
#         if execution_time > time_limit:
#             print('timeout')
#             break
#         if config["raise_error"] and epoch == 10:
#             raise Exception("unknown error")
        train_loss, train_acc, valid_loss, valid_acc = model.train(x_train, y_train, x_valid, y_valid, lr, n_epochs,
                                                                   batch_size)
        # log metric
#         wandb.log({
#             "loss/train_loss": train_loss,
#             "loss/valid_loss": valid_loss,
#             "accuracy/train_accuracy": train_acc,
#             "accuracy/valid_accuracy": valid_acc,
#             "params/epoch": epoch
#         })

    # evalute the model on test data
    Y_hat, _ = model.forward(x_test)
    test_loss = model.compute_loss(y_test, Y_hat)
    test_acc = model.evaluate(y_test, Y_hat)
    print("Final test loss = {:.3f}, acc = {:.3f}".format(test_loss, test_acc))
#     wandb.log({
#         "loss/test_loss": test_loss,
#         "accuracy/test_accuracy": test_acc
#     })
    threshold = 0.8
    # if test_acc < threshold:
#         wandb.alert(
#             title="low test accuracy ",
#             text=f"Accuracy {test_acc} is below the acceptable theshold {threshold}"
#         )



if __name__ == "__main__":
    with open('./config.json') as f:
        config = json.load(f)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", default = config["resume"], action = "store_true", help = "whether to resume")
    parser.add_argument("--run_id", default = config["run_id"], help = "run id to resume")
    parser.add_argument("--lr", type = float, default = config["lr"], help = "learning rate")
    parser.add_argument("--batch_size", type = int, default = config["batch_size"], help = "batch size")
    parser.add_argument("--raise_error", default = config["raise_error"], help = "batch size")
    parser.add_argument("--n_epochs", type = int, default = config["n_epochs"], help = "batch size")

    args = vars(parser.parse_args())
    main(args)
