# conda env create -f env.yml
# conda activate wandb-tutorial

import copy
import numpy as np
from mnist.data_utils import load_data
from tqdm import tqdm
import os
import wandb
import json
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy
import random
# Utils

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(784, 256)  # 5*5 from image dimension
        self.fc2 = nn.Linear(256, 10)
        # self.init_weight()
    def init_weight(self):
        self.fc1.weight.data.fill_(0.1)
        self.fc1.bias.data.fill_(0.1)
        self.fc2.weight.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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


def load_batch(X, Y, batch_size, shuffle=False):
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
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

# Load MNIST
def main(args):
    ### YOUR CODE HERE (3 lines)
    ### TODO:
    # Set random seed for python, numpy and torch (random seed = 133)
    


    # load json config file
    config = args
    # model instantiation
    model = Net()
    model.zero_grad()
    epoch_idx_global = 0
    accumulated_step = 0
    previous_step = 0
    train_loss = 0
    nested_break = False
    time_limit = config["time_limit"] # minutes
    lr, n_epochs, batch_size = config["lr"], config["n_epochs"], config["batch_size"]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99999999)
    # wandb instantiation
    ### YOUR CODE HERE (1 line)
    ### TODO:
    # sign in to wandb and activate on your shell
    # initialize wandb with following arguments:
    ## project: wandb-tutorial
    ## config: variable 'config' declared above
    ## notes: "Hello, wandb!"
    ## tags: ["tutorial"] 
    # wandb.init: https://docs.wandb.ai/ref/python/init


    # End your code
    model_output_path = config["save_dir"]+"/"+run.id
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    with open(os.path.join(model_output_path, "training_params.json"), 'w') as outfile:
        json.dump(config, outfile)
    start_time = time.time()
    # load and shuffle data
    x_train, y_train, x_test, y_test = load_data()
    idxs = np.arange(len(x_train))
    np.random.shuffle(idxs)
    split_idx = int(np.ceil(len(idxs) * 0.8))
    x_valid, y_valid = x_train[idxs[split_idx:]], y_train[idxs[split_idx:]]
    x_train, y_train = x_train[idxs[:split_idx]], y_train[idxs[:split_idx]]
 
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    x_valid = torch.tensor(x_valid)
    y_valid = torch.tensor(y_valid)
    train_tensor_data = TensorDataset(x_train, y_train)
    valid_tensor_data = TensorDataset(x_valid, y_valid)
    train_sampler = RandomSampler(train_tensor_data)
    train_dataloader = DataLoader(
        train_tensor_data, 
        num_workers=3,
        worker_init_fn=seed_worker,
        batch_size = config["batch_size"]
        )
    valid_sampler = RandomSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, 
        num_workers=3,
        worker_init_fn=seed_worker,
        batch_size = config["batch_size"]

        )
    print('Set validation data aside')
    print('Training data shape: ', x_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Validation data shape: ', x_valid.shape)
    print('Validation labels shape: ', y_valid.shape)

    # train the model"

    print("check hyper parameters")
    print("learning rate: ", lr)
    print("n_epochs: ", n_epochs)
    print("batch size: ", batch_size)
    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(epoch_idx_global, n_epochs)):
        step = 0
        train_loss = 0

        print("epoch:", epoch)
        execution_time = (time.time() - start_time) / 60

        print("execution time:", execution_time)
        if config["raise_error"] and epoch == 10:
            raise Exception("unknown error")
        for step, batch in enumerate(train_dataloader):
            step+=1
            execution_time = (time.time() - start_time) / 60
            if execution_time > time_limit:
                nested_break = True
                print('timeout')
                break
            if epoch <= epoch_idx_global and step <= previous_step:
              continue
            X_batch = batch[0]
            Y_batch = batch[1]
            optimizer.zero_grad()
            y_hat = model(X_batch)
            _, Y_batch = Y_batch.max(dim=1)
            train_loss = criterion(y_hat, Y_batch)
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            accumulated_step += 1
            if not step % config["save_interval"]:
                print("***** Saving fine - tuned model at {} *****".format(step))
                ### YOUR CODE HERE (3~7 lines)
                ### TODO:
                # Save your model
                # ToDo:
                # save your model to the path model_output_path/epoch_{epoch}_{step}. model_output_path is defined beforehand.
                # e.g. for the model at step 10 of epoch 1, your model is supposed to be saved in the directory './save/[run_id]/epoch_1_10'
                # use 'torch.save' to save your checkpoint. checkpoint file should include epoch, model.state_dict(), optimizer.state_dict(), step, and accumulated_step.
                # Do not delete this file. It will be used for the second assignment
                # torch save: https://pytorch.org/docs/stable/generated/torch.save.html


                # End your code
        if nested_break == True:
            break

        print("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
        model_output_path, "epochs/epoch_{}".format(epoch)
    )
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        }, epoch_output_folder_path)
        # track metric in wandb
        ### YOUR CODE HERE (~3 lines)
        ### TODO:
        # keep track of train_loss, epoch, and learning rate
        # wandb.log: https://docs.wandb.ai/ref/python/init
        # How to get current lr of optimizer with adaptive lr?: https://discuss.pytorch.org/t/get-current-lr-of-optimizer-with-adaptive-lr/24851/2

        # end your code





if __name__ == "__main__":
    with open('./config.json') as f:
        config = json.load(f)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", action = "store_true", help = "whether to resume")
    parser.add_argument("--run_id", type = str, help = "run id to resume")
    parser.add_argument("--lr", type = float, default =  0.01, help = "learning rate")
    parser.add_argument("--batch_size", type = int, default = 32, help = "batch size")
    parser.add_argument("--raise_error", action = "store_true", help = "raise error or not")
    parser.add_argument("--n_epochs", type = int, default = 10, help = "epochs")
    parser.add_argument("--save_interval", type = int, default = 100, help = "save interval")
    parser.add_argument("--save_dir", type = str, default = "./save", help = "save directory")
    parser.add_argument("--load_dir", type = str, default = "./save", help = "load directory")
    parser.add_argument("--time_limit", type = float, default = 1, help = "time limit in minutes")
    args = vars(parser.parse_args())
    main(args)
