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
#EPOCH = 1


#BATCH_SIZE = 256
#BATCH_SIZE = 128
BATCH_SIZE = 64
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
#input_data_folder = "../input/pmdpdsv14/pdsv14/pdsv14/"
#input_data_folder = "../input/pmdpdsv11/pdsv11/pdsv11/"
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

    #cnn = CNNv10().to(device)
    
    #cnn = resnet18().to(device)
    #cnn = resnet50().to(device)
    cnn = resnet101().to(device)
    print("\n cnn = ", cnn)
    
    #return


    train_loader, vali_loader = train_valid_loader(data_size=SMALL_DATA_SIZE, train_size=TRAIN_SIZE, num_workers=i_num_workers, pin_memory=b_pin_memory)
    
    
    
    #show_batch(train_loader, vali_loader)
    
    
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
    #cnn_test = CNNv10().to(device)
    #cnn_test = resnet18().to(device)
    #cnn_test = resnet50().to(device)
    cnn_test = resnet101().to(device)
    
    
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



""""the ResNet implementation is referenced from pytorch implementation, as following:"""
# for ResNet    

def conv3x3(i_in_planes, i_out_planes, i_stride=1):
    """3x3 convolution with padding"""
    
    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    return nn.Conv2d(in_channels=i_in_planes, out_channels=i_out_planes, kernel_size=3, stride=i_stride, padding=1, bias=False)


def conv1x1(i_in_planes, i_out_planes, i_stride=1):
    """1x1 convolution"""
    
    #return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    return nn.Conv2d(in_channels=i_in_planes, out_channels=i_out_planes, kernel_size=1, stride=i_stride, padding=0, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
        super(ResNet, self).__init__()
        
        self.inplanes = 32
        
        self.layer0 = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            #conv3x3(1, 16), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),    
        )
        
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        #self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        
        #self.bn1 = nn.BatchNorm2d(64)
        
        #self.relu = nn.ReLU(inplace=True)
        
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(256 * block.expansion, num_classes)
        )

        
        
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                #conv3x3(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        
        #print("before layer0: x.size", x.size())
        
        x = self.layer0(x)        
        
        #print("after layer0: x.size", x.size())
        
        x = self.layer1(x)
        
        #print("after layer1: x.size", x.size())
        
        x = self.layer2(x)        
        
        #print("after layer2: x.size", x.size())
        
        x = self.layer3(x)
                
        #print("after layer3: x.size", x.size())
        
        x = self.layer4(x)
        
        #print("after layer4: x.size", x.size())
        
        x = self.avgpool(x)
                
        #print("after avgpool: x.size", x.size())
        
        x = x.view(x.size(0), -1)
                
        #print("after view: x.size", x.size())
        
        x = self.classifier(x)
        
        #print("after classifier: x.size", x.size())
        
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model








if __name__ == '__main__':
    main()

 