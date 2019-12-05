
"""
script to train a resnet 50 network only with n epoch

rendering directly after each parameter estimation
"""
import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda
from utils_functions.MyResnet import Myresnet50
from utils_functions.MyResnet_t import Myresnet50_t
from utils_functions.train_test_val_V1 import training
from utils_functions.cubeDataset import CubeDataset
from datetime import date, datetime



xvalue = []
yvalue = []
zvalue = []
allval = []


filepath = 'value2plot.txt'
with open(filepath) as fp:
   line = fp.readline()
   cnt = 1
   while line:
       text = line.strip()
       print("Line {}: {}".format(cnt, line.strip()))


       if cnt != 1 and cnt < 163:
            split = line.split()
            xvalue.append(float(split[0]))
            yvalue.append(float(split[1]))
            zvalue.append(float(split[2]))
            print('split')


       line = fp.readline()
       cnt += 1

# print(len(xvalue))

plt.plot(xvalue)
plt.plot(yvalue)
plt.plot(zvalue)
plt.title('Translation Error')
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.yscale('log')
plt.legend(['x-axis', 'y-axis', 'z-axis'])
plt.show()
