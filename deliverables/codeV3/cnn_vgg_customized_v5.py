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
from torch.autograd import Variable

import matplotlib.pyplot as plt

import time
# print(os.listdir("Data"))

EPOCH = 100
#EPOCH = 50
#EPOCH = 10
#EPOCH = 5


#BATCH_SIZE = 256
BATCH_SIZE = 128
#BATCH_SIZE = 64
#BATCH_SIZE = 32

LR = 0.0002
#TRAIN_SIZE = 0.9
TRAIN_SIZE = 0.8
#TRAIN_SIZE = 0.2

SMALL_DATA_SIZE = 1
# VALI_SIZE = 0.05

i_image_size = 64

b_dl_new_or_old = False


str_folder = "../input/comp-551-w2019-project-3-modified-mnist/"


input_data_folder = "../input/pmdpdsv6/pdsv6/pdsv6/"
#input_data_folder = "../input/pmdpdsv5/pdsv5/pdsv5/"
#input_data_folder = "../input/pmdpdsv7/pdsv7/pdsv7/"
#input_data_folder = "../input/preprocessed-mmnist-data/pds/pds/"

str_file_version_postfix = "_v6"
#str_file_version_postfix = ""


print("input_data_folder = ", input_data_folder)

#str_train_file = 'train_w_tidy' + str_file_version_postfix + '.csv'
str_train_file = 'train' + str_file_version_postfix + '.csv'
str_test_file = 'test' + str_file_version_postfix + '.csv'


print(str_train_file)
print(str_test_file)



print(os.listdir("../input/"))

b_time_estimation = False

b_use_cuda = torch.cuda.is_available()

print("b_use_cuda = ", b_use_cuda)

#b_use_cuda = False

b_pin_memory = False
 
i_num_workers = 0

device = None

if b_use_cuda:
    device = torch.device('cuda')

    i_num_workers = 4
    
    b_pin_memory = True
    #b_pin_memory = False

else:
    device = torch.device('cpu')
    
    i_num_workers = 4
    
    b_pin_memory = False



total_m_b1_time = 0

total_m_b11_time = 0
total_m_b12_time = 0
total_m_b13_time = 0
total_m_b14_time = 0
total_m_b15_time = 0
total_m_b16_time = 0
total_m_b17_time = 0
total_m_b18_time = 0

total_m_b2_time = 0

total_m_b21_time = 0
total_m_b22_time = 0
total_m_b23_time = 0


total_m_b3_time = 0
total_m_b4_time = 0


def main():

    #use_cuda = torch.cuda.is_available()

    #print(use_cuda)

    #device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, vali_loader = train_valid_loader(data_size=SMALL_DATA_SIZE, train_size=TRAIN_SIZE, num_workers=i_num_workers, pin_memory=b_pin_memory)
    
    
    
    #show_batch(train_loader, vali_loader)
    cnn = CNNv9().to(device)
    #print("\n cnn = ", cnn, cnn.device)
    
    
    #print
    
    
    
    
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    
    #print("\n optimizer = ", optimizer, optimizer.device)
    
    
    #quit()
    
    loss_func = nn.CrossEntropyLoss()
    CNN_parameter = cnn.state_dict()
    best_accuracy = 0.
    for epoch in range(EPOCH):
        
        
        
        st_l = 0
        et_l = 0
        
        st_l = time.time()
        
        cnn.train()
        for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            # print(type(b_x))
            # print(b_x, b_y)
            #print("\n training step = ", step, b_x.size(),  b_y.size())
            #print("\n types = ", type(b_x), type(b_y))
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            et_l = time.time()
            ut = et_l - st_l
            global total_m_b3_time
            total_m_b3_time += ut
			
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            st_b = time.time()
            
            st = time.time()
            
            
            
            optimizer.zero_grad()           # clear gradients for this training step
            
            tmp_view = b_x.view(-1, 1, i_image_size, i_image_size)
            trains = Variable(tmp_view)
            labels = Variable(b_y)
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            et = time.time()
            ut = et - st
            global total_m_b18_time
            total_m_b18_time += ut
            
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            st = time.time()
            
            #b_x, b_y = b_x.to(device), b_y.to(device)
            
            
            trains = trains.to(device)
            # = b_x.to(device)
            #b_x = b_x.cuda()
			
			
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            et = time.time()
            ut = et - st
            global total_m_b16_time
            total_m_b16_time += ut
			
			
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            st = time.time()
			
            labels = labels.to(device)
            #b_y = b_y.to(device)
			
			
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
			            
            et = time.time()
            ut = et - st
            global total_m_b17_time
            total_m_b17_time += ut
            
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            st = time.time()
            
            
            output = cnn(trains)
            
            #output = cnn(b_x)
            # print(output.size())
            # if(epoch == 0 and step == 45):
            #     print(output, b_y)
            # print(b_y.size())            # cnn output
            
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            
            et = time.time()
            ut = et - st
            global total_m_b11_time
            total_m_b11_time += ut
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            st = time.time()
            
            
            #loss = loss_func(output, b_y)   # cross entropy loss
            loss = loss_func(output, labels) 
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
                
            et = time.time()
            ut = et - st
            global total_m_b12_time
            total_m_b12_time += ut
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            st = time.time()
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            et = time.time()
            ut = et - st
            global total_m_b13_time
            total_m_b13_time += ut
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            st = time.time()
            
            
            loss.backward()                 # backpropagation, compute gradients
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            et = time.time()
            ut = et - st
            global total_m_b14_time
            total_m_b14_time += ut
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            st = time.time()
            
            
            optimizer.step()                # apply gradients
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            et = time.time()
            ut = et - st
            global total_m_b15_time
            total_m_b15_time += ut
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            et_b = time.time()
            ut = et_b - st_b
            global total_m_b1_time
            total_m_b1_time += ut
            
            if b_use_cuda and b_time_estimation:
                torch.cuda.synchronize()
            
            st_l = time.time()

        if True:
            # print(output, b_y)
            right_num = list()
            total_num = list()
            
            
        
            st_l = time.time()
            cnn.eval()
            
            for step_v, (vali_x, vali_y) in enumerate(vali_loader):
                
                #print("\n validation step_v = ", step_v, vali_x.size(),  vali_y.size())
                #print("\n types = ", type(vali_x), type(vali_y))
                
                if b_use_cuda and b_time_estimation:
                    torch.cuda.synchronize()
                
                et_l = time.time()
                ut = et_l - st_l
                global total_m_b4_time
                total_m_b4_time += ut
                
                if b_use_cuda and b_time_estimation:
                    torch.cuda.synchronize()
                
                st_b = time.time()
            
                
                st = time.time()
            
                vali_x = vali_x.to(device)
                
                if b_use_cuda and b_time_estimation:
                    torch.cuda.synchronize()
                
                et = time.time()
                ut = et - st
                global total_m_b23_time
                total_m_b23_time += ut
                
                
                if b_use_cuda and b_time_estimation:
                    torch.cuda.synchronize()
                
                st = time.time()
                
                #cnn.eval()
                vali_output = cnn(vali_x)
            
                if b_use_cuda and b_time_estimation:
                    torch.cuda.synchronize()
                
                et = time.time()
                ut = et - st
                global total_m_b21_time
                total_m_b21_time += ut
                
                if b_use_cuda and b_time_estimation:
                    torch.cuda.synchronize()
                
                st = time.time()
                
            
                # print(vali_output.size())
                pred_y = torch.max(vali_output, 1)[1].cpu().data.numpy()
                #pred_y = torch.max(vali_output, 1)[1].cpu()
                # print(pred_y.shape)
                # print(float((pred_y == vali_y.numpy()).sum()), len(pred_y))
                # accuracy = float((pred_y == vali_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                
                if b_use_cuda and b_time_estimation:
                    torch.cuda.synchronize()
                
                et = time.time()
                ut = et - st
                global total_m_b22_time
                total_m_b22_time += ut
                
                
                right_num.append(float((pred_y == vali_y.numpy()).sum()))
                total_num.append(float(len(pred_y)))
                
                if b_use_cuda and b_time_estimation:
                    torch.cuda.synchronize()
                
                et_b = time.time()
                ut = et_b - st_b
                global total_m_b2_time
                total_m_b2_time += ut
                
                if b_use_cuda and b_time_estimation:
                    torch.cuda.synchronize()
                
                st_l = time.time()
                
            accuracy = sum(right_num) / sum(total_num)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                CNN_parameter = cnn.state_dict()

            print('Epoch: ', epoch, '| train loss: %.8f' % loss.cpu().data.numpy(), '| test accuracy: %.8f' % accuracy, 'Total right: ', sum(right_num), '| Total: ', sum(total_num))
    
    torch.save(CNN_parameter, 'cnn_para.pkl')

    print("\n best_accuracy = ", best_accuracy)
    # cnn_test = cnn_test.to(device)
    test(CNN_parameter, device)
    
    
    
    print("\n\n total_m_b1_time = ", total_m_b1_time)
    
    
    print("\n total_m_b11_time = ", total_m_b11_time)
    
    print("\n total_m_b12_time = ", total_m_b12_time)
    
    print("\n total_m_b13_time = ", total_m_b13_time)
    
    print("\n total_m_b14_time = ", total_m_b14_time)
    
    print("\n total_m_b15_time = ", total_m_b15_time)
    
    print("\n total_m_b16_time = ", total_m_b16_time)
    
    print("\n total_m_b17_time = ", total_m_b17_time)
	
    print("\n total_m_b18_time = ", total_m_b18_time)
    
    
    print("\n\n total_m_b2_time = ", total_m_b2_time)
    
    print("\n total_m_b21_time = ", total_m_b21_time)
    
    print("\n total_m_b22_time = ", total_m_b22_time)
    
    print("\n total_m_b23_time = ", total_m_b23_time)
    
    
    print("\n\n total_m_b3_time = ", total_m_b3_time)
    
    print("\n\n total_m_b4_time = ", total_m_b4_time)


def test(CNN_parameter, device):
    cnn_test = CNNv9().to(device)
    cnn_test.load_state_dict(CNN_parameter)
    print('a')
    
    test_images = None
    if b_dl_new_or_old:
    
        test = pd.read_csv(input_data_folder + str_test_file)
    
        test_images = test.iloc[:, 0:]
        
        test_images = test_images.values
        
        test_images = np.reshape(test_images, (test_images.shape[0], 64, 64))
        
    else:
        test_images = pd.read_pickle(str_folder + 'test_images.pkl')    
    
    
    print("test_images debug:", type(test_images), test_images.shape, test_images[2])
    
    X = torch.tensor(test_images / 255.)
    X = torch.unsqueeze(X, dim=1).type(torch.FloatTensor)
    pred_y = list()
    
    cnn_test.eval()
    for x in X:
        x = x.view(1, 1, 64, 64)
        x = x.to(device)        
        test_output = cnn_test(x)
        pred_y.append(torch.max(test_output, 1)[1].cpu().data.numpy()[0])
    prediction_print(pred_y)


def prediction_print(y_test):
    count = 0
    
    cur_time = int(time.time())
    
    csv_fn = "prediction_cnn_" + str(cur_time) + ".csv"
    
    print("\n csv_fn = ", csv_fn)
    
    with open(csv_fn, 'w', newline='', encoding='utf-8') as csv_file:
        print("Printing prediction to csv file... ")
        writer = csv.writer(csv_file)
        writer.writerow(['ID', 'Category'])
        for i in y_test:
            writer.writerow([count, i])
            count += 1

    csv_file.close()


def show_batch(train_loader, vali_loader):
    for loader_item in (train_loader, vali_loader):   # train entire dataset 3 times
        for step, (batch_x, batch_y) in enumerate(loader_item):  # for each training step
            # train your data...
            print('loader: ', loader_item, '| Step: ', step, '| batch x: ',
                  batch_x.size(), '| batch y: ', batch_y.size())


def train_valid_loader(data_size=1, train_size=0.8,
                       num_workers=1,
                       pin_memory=True):

    train = None
    data_images = None
    
    if b_dl_new_or_old:
        
        train = pd.read_csv(input_data_folder + str_train_file)
    
        data_images = train.iloc[:, 1:]
        
        data_images = data_images.values
        
        data_images = np.reshape(data_images, (data_images.shape[0], 64, 64))
        
    else:
        
        data_images = pd.read_pickle(str_folder + 'train_images.pkl')
    
    
    
    print("data_images debug:", type(data_images), data_images.shape, data_images[2])
    #print("data_images debug:", type(data_images), data_images[2])
    
    # train_data = torch.from_numpy(train_images)
    X = torch.tensor(data_images / 255.)
    X = X[:int(X.size(0) * data_size)]
    X = torch.unsqueeze(X, dim=1).type(torch.FloatTensor)
    print(X.size())

    data_labels = None
    
    if b_dl_new_or_old:
        data_labels = train.iloc[:, 0]
        
        data_labels = data_labels.values
        
    else:
        data_labels_raw = pd.read_csv(str_folder + 'train_labels.csv')
        data_labels = data_labels_raw['Category']
    
    

    #print("data_labels debug:", type(data_labels), data_labels[2])
    print("data_labels debug:", type(data_labels), data_labels.shape, data_labels[2])
      
    
    #y = torch.tensor(data_labels['Category'])
    y = torch.tensor(data_labels)
    
    
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



#follow the VGG principle



# try normalize for conv1, compared to v7
class CNNv9(nn.Module):
    def __init__(self):
        super(CNNv9, self).__init__()
         
        self.conv1 = nn.Sequential( #W0 = 64
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0), #W1 = 60
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1), #W2 = 60
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=2, stride=1, padding=0), #W2 = 59
            #nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), #W3 = 29
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), #W4 = 29
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1), #W5 = 29
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), #W6 = 14
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),  #W7 = 14
            #nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),  #W8 = 14
            #nn.BatchNorm2d(512),
            nn.ReLU(),                 # activation
            nn.MaxPool2d(kernel_size=2, stride=2),  # W9 = 7
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),  #W10 = 7
            #nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),  #W11 = 7
            #nn.BatchNorm2d(512),
            nn.ReLU(),                 # activation
            nn.MaxPool2d(kernel_size=3, stride=2),  # W12 = 3
        )
        
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),  #W13 = 3
            #nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),  #W14 = 3
            #nn.BatchNorm2d(512),            
            nn.ReLU(),                 # activation
            nn.MaxPool2d(kernel_size=3, stride=2),  # W15 = 1
        )
        
        """
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  #W13 = 2
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  #W14 = 2
            nn.ReLU(),                 # activation
            #nn.MaxPool2d(kernel_size=2, stride=2),  # W15 = 2
        )
        """
        
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(1024, 256),
            #nn.Linear(1152 * 3 * 3, 128),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(256, 10)
        )


    def forward(self, x):
        x = self.conv1(x)
        # print('a')
        x = self.conv2(x)
        # print('a')
        x = self.conv3(x)
        # print('a')
        x = self.conv4(x)
        
        x = self.conv5(x)
        
        #x = self.conv6(x)
        
        x = x.view(x.size(0), -1)           # flatten the output of conv2
        #x = self.linear1(x)
        # x = self.linear2(x)
        output = self.classifier(x)
        return output    # return x for visualization









if __name__ == '__main__':
    main()

 