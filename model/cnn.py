from __future__ import print_function, division

from utils import plot_functions

from PIL import Image

import os
import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import time
import matplotlib.pyplot as plt
import seaborn


def pil_loader(path):
    """
    Load an image into PIL format and convert it into RGB
    :param path: String, Complete path of the image file
    :return: PIL image
    """
    image = Image.open(path)
    return image.convert("RGB")


def show_tensor_image(tensor):
    """
    Take a tensor and show the corresponding image
    :param tensor: Pytorch Tensor, [channels, height, width]
    :return:
    """
    tensor = tensor.transpose(0, 1)
    tensor = tensor.transpose(1, 2)
    io.imshow(tensor.cpu().numpy())


def create_valid_train_set(csv_info_name, data_clean_dir, dataset_name, test_size):
    csv_info_name = os.path.join(data_clean_dir, dataset_name, csv_info_name)
    csv_info = pd.read_csv(csv_info_name)
    tmp = csv_info[["record_id", "tag"]].drop_duplicates()
    X_train, X_valid = train_test_split(tmp, test_size=test_size,
                                        random_state=42, stratify=tmp["tag"])
    X_train = X_train[["record_id"]].merge(csv_info, on="record_id")
    X_valid = X_valid[["record_id"]].merge(csv_info, on="record_id")

    X_train_tmp = X_train[X_train["tag"] != "idle"]
    X_train = pd.concat([X_train, *([X_train_tmp] * 4)], axis=0)
    return X_train, X_valid


class SatelliteImageDataset(Dataset):
    """Load a satellite dataset"""

    def __init__(self, X, transform=transforms.ToTensor(), device=torch.device("cpu")):
        """
        Create a satellite image dataset
        :param transform: torchvion transform function, Optional transform to be applied
                on an image.
        :device: Pytorch device: cpu or gpu to move the data into the good device
        """
        self.X = X
        self.L_image_path = list(self.X["image_clean_path"])
        lab_enc = LabelEncoder()
        self.X["tag"] = lab_enc.fit_transform(self.X["tag"])
        self.L_tag = list(self.X["tag"])
        self.classes_ = lab_enc.classes_
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.L_image_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = pil_loader(self.L_image_path[idx])
        image = self.transform(image)
        return image.to(self.device), torch.tensor(self.L_tag[idx]).to(self.device)


class Net(nn.Module):
    def __init__(self, input_size=3, output_size=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 29 * 29)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def trainNet(net, batch_size, n_epochs, learning_rate, train_dataset, valid_dataset, num_workers, path_model, net_name):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    # Get training data
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              num_workers=num_workers, shuffle=True)
    n_batches = len(train_loader)

    # Loss function
    loss = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Time for printing
    training_start_time = time.time()

    best_valid_loss = np.inf

    L_loss_train = []
    L_loss_valid = []

    # Loop for n_epochs
    for epoch in range(n_epochs):

        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):
            # Get inputs
            inputs, labels = data

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss_size.detach().item()
            total_train_loss += loss_size.detach().item()

            # Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in valid_loader:
            # Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.detach().item()

        L_loss_train.append(total_train_loss / len(train_loader))
        L_loss_valid.append(total_val_loss / len(valid_loader))

        if total_val_loss < best_valid_loss:
            best_valid_loss = total_val_loss
            torch.save(net.state_dict(), os.path.join(path_model, net_name))

        print("Validation loss = {:.2f}".format(total_val_loss / len(valid_loader)))

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    return L_loss_train, L_loss_valid


def predict(net, X):
    """
    return the prediction of the read2genome algorithm
    :param X: Tensor [batch_size, n_dim], matrix
    :return: 1-D torch Tensor
    """
    y = torch.nn.Softmax(dim=1)(net.eval()(X))
    return torch.argmax(y, dim=1), y


def compute_score(net, dataloader):
    """Return the classification score rate for a dataset
    in a dataloader"""
    with torch.no_grad():
        nb_elem = 0
        score = 0
        L_y_true = []
        L_y_pred = []
        L_y_softmax = []
        for it, (X, y_) in enumerate(iter(dataloader)):
            y, y_softmax = predict(net, X)
            L_y_true += list(y_.cpu().numpy())
            L_y_pred += list(y.cpu().numpy())
            L_y_softmax += list(y_softmax.cpu().numpy())
            nb_elem += len(X)
            score += (y.int() == y_.int()).sum().item()
        return score * 1. / nb_elem, np.array(L_y_true), np.array(L_y_pred), np.array(L_y_softmax)


if __name__ == "__main__":
    data_dir = "/classification_speed_boat/data/"
    data_clean_dir = "/classification_speed_boat/data_clean/"
    dataset_name = "train"
    csv_clean_name = "info_boat.csv"

    model_name = "cnn3"
    net_name = "cnn_basic.pt"
    prediction_dir = "/classification_speed_boat/prediction/"
    path_model = os.path.join(prediction_dir, model_name)
    if not os.path.exists(path_model):
        os.makedirs(path_model)

    # Parameters
    test_size = 0.2

    size = 128
    transform = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.RandomRotation((0, 180)),
                                    transforms.ToTensor(),
                                    transforms.RandomErasing()])
    transform_valid = transforms.Compose([transforms.Resize((size, size)),
                                          transforms.ToTensor()])
    batch_size = 64
    num_workers = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_train, X_valid = create_valid_train_set(csv_clean_name, data_clean_dir, dataset_name, test_size)

    # Create dataset
    train_dataset = SatelliteImageDataset(X_train, transform, device=device)
    valid_dataset = SatelliteImageDataset(X_valid, transform_valid, device=device)

    input_size = 3
    output_size = 3

    n_epochs = 50
    learning_rate = 0.0005

    cnn = Net(input_size, output_size)
    cnn = cnn.to(device)

    L_loss_train, L_loss_valid = trainNet(cnn, batch_size, n_epochs, learning_rate, train_dataset, valid_dataset, num_workers, path_model, net_name)
    df = pd.DataFrame(np.array([L_loss_valid, L_loss_train]).T, columns=["validation_loss", "train_loss"])
    fig = df.plot().get_figure()
    fig.savefig(os.path.join(path_model, "train_val_loss.png"))

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              num_workers=num_workers, shuffle=True)

    score, y_true, y_pred, y_softmax = compute_score(cnn, valid_loader)
    print("Validation score : %s\n" % score)
    ax = plot_functions.plot_confusion_matrix(y_true, y_pred, train_dataset.classes_,
                                 cmap=plt.cm.Blues)
    ax.get_figure().savefig(os.path.join(path_model, "confusion_matrix.png"))
