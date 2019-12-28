#!/usr/bin/env python3

import matplotlib.pyplot as plt
import glob
import os
import sys
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.autograd import Variable
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import argparse

def scale(X, x_min, x_max):
    imgs, rgb, d_x, d_y = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    if rgb != 3:
        print("Input shape of image error, please input images with dimensions: images*rgb*x*y")
    else:
        images_reshape = X.reshape(imgs, rgb, (d_x * d_y))
        X_min = images_reshape.min(axis=2)
        X_min_expand_axis = np.repeat(X_min[:,:,np.newaxis], (d_x * d_y), axis = 2)
        X_max = images_reshape.max(axis=2)
        X_max_expand_axis = np.repeat(X_max[:,:,np.newaxis], (d_x * d_y), axis = 2)
        X_range = X_max_expand_axis - X_min_expand_axis
        image_normal = (x_max-x_min)/X_range * (images_reshape - X_min_expand_axis) + (x_min)
        image_normal_reshape = image_normal.reshape(imgs, rgb, d_x, d_y)
        return image_normal_reshape
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=8):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(3, stride=1)
        self.fc = nn.Linear(512 * 9 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model

if __name__ == '__main__':
    
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="please give the input histo image with .tif")
    parser.add_argument("-r", "--resolution", type=int, default=30, help="please set the stride for prediction")
    #parser.add_argument("-o", "--output", help="please give the ouput name of filtered image with .png")
    args = parser.parse_args()
    
    input_file = args.input
    #output_file = args.output
    output_file_image = input_file + "_result.png"
    output_file_description = input_file + "_result.txt"
    
    # Read image
    histo_image = plt.imread(input_file)
    
    # cut images and prediction
    net_input_pixel = 150
    stride = args.resolution # could be revised to improve resolution
    
    x_part = (histo_image.shape[0] - net_input_pixel) // stride
    x_space = ((histo_image.shape[0] - net_input_pixel) - stride * (x_part - 1)) // 2
    y_part = (histo_image.shape[1] - net_input_pixel) // stride
    y_space = ((histo_image.shape[1] - net_input_pixel) - stride * (y_part - 1)) // 2
    total = int(x_part * y_part)
    
    # Read model
    resnet = resnet34()
    resnet.load_state_dict(torch.load("HISTO_MODEL_RESNET34_ver12_rescale_four_fold3.pkl"))
    
    # Start prediction
    count = 1
    histo_matrix_label = []
    
    for i in range(x_part):
        histo_matrix_x_label = []
        
        for j in range(y_part):
            x_start = int(i * stride + x_space)
            x_end = int(x_start + net_input_pixel)
            y_start = int(j * stride + y_space)
            y_end = int(y_start + net_input_pixel)

            histo_sub = histo_image[x_start:x_end,y_start:y_end,:]

            histo_sub = histo_sub[np.newaxis,:,:,:]
            histo_sub = np.moveaxis(histo_sub, -1, 1)

            histo_sub_normal = scale(histo_sub, -1, 1)
            train_torch = torch.from_numpy(histo_sub_normal).float()

            prediction = resnet(train_torch)
            prediction_final = torch.max(prediction, 1)[1].data.squeeze()

            histo_matrix_x_label.append(prediction_final.numpy().item())

            processing = "Prediction..." + str(round(((count)/ total * 100),2)) + "% " +"\r"
            sys.stdout.write(processing)
            sys.stdout.flush()
            count += 1

        histo_matrix_label.append(histo_matrix_x_label)

    histo_matrix_label = np.array(histo_matrix_label)
    
    # Result description
    histo_matrix_label_f = histo_matrix_label.flatten()
    fp = open(output_file_description, "a")
    fp.write("Tumor: "+ str(sum(histo_matrix_label_f == 0)) + "\n")
    fp.write("Stroma: "+ str(sum(histo_matrix_label_f == 1)) + "\n")
    fp.write("Complex: "+ str(sum(histo_matrix_label_f == 2)) + "\n")
    fp.write("Lympho: "+ str(sum(histo_matrix_label_f == 3)) + "\n")
    fp.write("Debris: "+ str(sum(histo_matrix_label_f == 4)) + "\n")
    fp.write("Mucosa: "+ str(sum(histo_matrix_label_f == 5)) + "\n")
    fp.write("Adipose: "+ str(sum(histo_matrix_label_f == 6)) + "\n")
    fp.write("Empty: "+ str(sum(histo_matrix_label_f == 7)) + "\n")
    if sum(histo_matrix_label_f == 1) != 0:
        fp.write("Tumor/Stroma ratio = "+ str(sum(histo_matrix_label_f == 0)/sum(histo_matrix_label_f == 1)) + "\n")
    fp.close()
    
    print("=========Summary Report=========")
    print("Tumor: ", str(sum(histo_matrix_label_f == 0)))
    print("Stroma: ", str(sum(histo_matrix_label_f == 1)))
    print("Complex: ", str(sum(histo_matrix_label_f == 2)))
    print("Lympho: ", str(sum(histo_matrix_label_f == 3)))
    print("Debris: ", str(sum(histo_matrix_label_f == 4)))
    print("Mucosa: ", str(sum(histo_matrix_label_f == 5)))
    print("Adipose: ", str(sum(histo_matrix_label_f == 6)))
    print("Empty: ", str(sum(histo_matrix_label_f == 7)))
    if sum(histo_matrix_label_f == 1) != 0:
        print("Tumor/Stroma ratio = ", str(sum(histo_matrix_label_f == 0)/sum(histo_matrix_label_f == 1)))
    
    # Plot result
    plt.figure(figsize = (10,7))
    sns.heatmap(histo_matrix_label, annot=False, annot_kws={"size": 10}, fmt='g')
    plt.savefig(output_file_image)
    print("================================")
    
    