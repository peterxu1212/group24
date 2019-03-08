import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# ACCURACY 0.94-0.95
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
import csv
# third-party library
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
# print(os.listdir("Data"))
EPOCH = 50
BATCH_SIZE = 256
LR = 0.0002
TRAIN_SIZE = 0.9
SMALL_DATA_SIZE = 1
# VALI_SIZE = 0.05


def main():

    use_cuda = torch.cuda.is_available()

    print(use_cuda)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, vali_loader = train_valid_loader(data_size=SMALL_DATA_SIZE, train_size=TRAIN_SIZE)
    # show_batch(train_loader, vali_loader)
    cnn = CNN().to(device)
#     # print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()
    CNN_parameter = cnn.state_dict()
    best_accuracy = 0.
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            # print(type(b_x))
            # print(b_x, b_y)
            # print(b_x.size(), b_y.size())
            b_x, b_y = b_x.to(device), b_y.to(device)
            output = cnn(b_x)
            # print(output.size())
            # if(epoch == 0 and step == 45):
            #     print(output, b_y)
            # print(b_y.size())            # cnn output
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

        if True:
            # print(output, b_y)
            right_num = list()
            total_num = list()
            for step_v, (vali_x, vali_y) in enumerate(vali_loader):
                vali_x = vali_x.to(device)
                vali_output = cnn(vali_x)
                # print(vali_output.size())
                pred_y = torch.max(vali_output, 1)[1].cpu().data.numpy()
                # print(pred_y.shape)
                # print(float((pred_y == vali_y.numpy()).sum()), len(pred_y))
                # accuracy = float((pred_y == vali_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                right_num.append(float((pred_y == vali_y.numpy()).sum()))
                total_num.append(float(len(pred_y)))
            accuracy = sum(right_num) / sum(total_num)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                CNN_parameter = cnn.state_dict()

            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.2f' % accuracy, 'Total right: ', sum(right_num), '| Total: ', sum(total_num))
    torch.save(CNN_parameter, 'cnn_para.pkl')

    # cnn_test = cnn_test.to(device)
    test(CNN_parameter, device)


def test(CNN_parameter, device):
    cnn_test = CNN().to(device)
    cnn_test.load_state_dict(CNN_parameter)
    print('a')
    test_images = pd.read_pickle('Data/test_images.pkl')
    X = torch.tensor(test_images / 255.)
    X = torch.unsqueeze(X, dim=1).type(torch.FloatTensor)
    pred_y = list()
    for x in X:
        x = x.view(1, 1, 64, 64)
        x = x.to(device)
        test_output = cnn_test(x)
        pred_y.append(torch.max(test_output, 1)[1].cpu().data.numpy()[0])
    prediction_print(pred_y)


def prediction_print(y_test):
    count = 0
    with open('prediction_svm.csv', 'w', newline='', encoding='utf-8') as csv_file:
        print("Printing prediction to csv file... ")
        writer = csv.writer(csv_file)
        writer.writerow(['ID', 'Category'])
        for i in y_test:
            writer.writerow([count, i])
            count += 1

    csv_file.close()


def show_batch(train_loader, vali_loader):
    for epoch in range(1):   # train entire dataset 3 times
        for step, (batch_x, batch_y) in enumerate(train_loader):  # for each training step
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


def train_valid_loader(data_size=1, train_size=0.8,
                       num_workers=1,
                       pin_memory=True):

    data_images = pd.read_pickle('Data/train_images.pkl')
    # train_data = torch.from_numpy(train_images)
    X = torch.tensor(data_images / 255.)
    X = X[:int(X.size(0) * data_size)]
    X = torch.unsqueeze(X, dim=1).type(torch.FloatTensor)
    print(X.size())

    data_labels = pd.read_csv('Data/train_labels.csv')
    y = torch.tensor(data_labels['Category'])
    y = y[:int(y.size(0) * data_size)]
    print(y.size())

    torch_dataset = Data.TensorDataset(X, y)

    print(len(torch_dataset))
    train_dataset, vali_dataset = Data.random_split(torch_dataset, [int(len(torch_dataset) * train_size), len(torch_dataset) - int(len(torch_dataset) * train_size)])
    print(len(train_dataset), ' ', len(vali_dataset))

    train_loader = Data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    vali_loader = Data.DataLoader(
        vali_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, vali_loader)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 64, 64)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # W2=(W1-F+2P)/S+1=(64-F+2P)/S+1
            ),                              # output shape
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),            # activation
            nn.MaxPool2d(kernel_size=2, stride=2),    # W3=(W2-F)/S+1
        )
        self.conv2 = nn.Sequential(         # input shape
            nn.Conv2d(32, 64, 3, 1, 1),     # output shape (64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),                 # activation
            nn.MaxPool2d(kernel_size=2, stride=2),  #
        )
        self.conv3 = nn.Sequential(         # input shape
            nn.Conv2d(128, 256, 3, 1, 1),     # output shape (64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),                 # activation
            nn.MaxPool2d(kernel_size=2, stride=2),  #
        )

        self.conv4 = nn.Sequential(         # input shape
            nn.Conv2d(512, 512 + 256, 3, 1, 1),     # output shape (64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(512 + 256, 1024, 3, 1, 1),
            nn.ReLU(),                 # activation
            nn.MaxPool2d(kernel_size=2, stride=2),  #
        )

        self.linear1 = nn.Sequential(nn.Linear(1024 * 4 * 4, 1024 * 4), nn.ReLU(),)
        # self.linear2 = nn.Sequential(nn.Linear(1024 * 4, 1024 * 4), nn.ReLU(),)
        self.out = nn.Linear(1024 * 4, 10)

        # self.conv3 = nn.Sequential(         # input shape (64, 16, 16)
        #     nn.Conv2d(128, 256, 3, 1, 1),     # output shape (128, 16, 16)
        #     nn.ReLU(),                      # activation
        #     nn.MaxPool2d(kernel_size=2, stride=2),                # output shape (128, 8, 8)
        # )

        # self.out = nn.Linear(256 * 8 * 8, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        # print('a')
        x = self.conv2(x)
        # print('a')
        x = self.conv3(x)
        # print('a')
        x = self.conv4(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2
        x = self.linear1(x)
        # x = self.linear2(x)
        output = self.out(x)
        return output    # return x for visualization


if __name__ == '__main__':
    main()

 # First try 78% vali
 # self.conv1 = nn.Sequential(
 #            nn.Conv2d(1,64,5,1,2),
 #            nn.ReLU(),
 #            nn.MaxPool2d(kernel_size=8),
 #        )
 #        self.conv2 = nn.Sequential(
 #            nn.Conv2d(64, 256, 5, 1, 2),
 #            nn.ReLU(),
 #            nn.MaxPool2d(8),
 #        )
 #        self.out = nn.Linear(256 * 1 * 1, 10)
