import json
import torch
import argparse
import os
import numpy as np
# import pysilico
import json
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import imageio

with open('dictSave/testsave1100New.json') as json_file:
    data = json.load(json_file)
    data_len = len(data)
    # usm_camera = data[0:data_len]['usm-1']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)

file_name_extension = '444_images3'  # choose the corresponding database to use
file_name_extension_validation = '693_images2'  # choose the corresponding database to use
ShaftSetName = 'Shaft_{}'.format(file_name_extension) #used to describe the document name

batch_size = 2
vallen = 100
n_epochs = 10



Background_file = 'Npydatabase/endoscIm_{}.npy'.format(file_name_extension)
RGBshaft_file = 'Npydatabase/RGBShaft_{}.npy'.format(file_name_extension)
BWShaft_file = 'Npydatabase/BWShaft_{}.npy'.format(file_name_extension)
parameters_file = 'Npydatabase/params_{}.npy'.format(file_name_extension)

Background_Valfile = 'Npydatabase/endoscIm_{}.npy'.format(file_name_extension_validation)
RGBshaft_Valfile = 'Npydatabase/RGBShaft_{}.npy'.format(file_name_extension_validation)
BWShaft_Valfile = 'Npydatabase/BWShaft_{}.npy'.format(file_name_extension_validation)
parameters_Valfile = 'Npydatabase/params_{}.npy'.format(file_name_extension_validation)


date4File = '15111' #mmddyy
obj_name = 'LongShaft2' #'shaftshortOnly'
comment = 'test'
type= 'render'
fileExtension = '{}{}_{}epochs_{}'.format(date4File,type, n_epochs,comment) #string to ad at the end of the file

Background = np.load(Background_file)
sils = np.load(BWShaft_file)
params = np.load(parameters_file)

print('param loaded')


train_im = Background[:]  # 90% training
train_sil = sils[:]
train_param = params[:]
paramX = params[:, 3]
paramY = params[:, 4]
paramZ = params[:, 5]
plt.hist(paramY, bins=20)
plt.show()