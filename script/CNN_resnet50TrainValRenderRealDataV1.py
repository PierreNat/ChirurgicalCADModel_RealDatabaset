
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
from utils_functions.train_val_regV3 import train_regV3
from utils_functions.train_val_renderV3 import train_renderV3
from utils_functions.cubeDataset import CubeDataset


# device = torch.device('cpu')
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
obj_name = 'LongShaft' #'shaftshortOnly'
comment = 'test'
type= 'render'
fileExtension = '{}{}_{}epochs_{}'.format(date4File,type, n_epochs,comment) #string to ad at the end of the file

Background = np.load(Background_file)
sils = np.load(BWShaft_file)
params = np.load(parameters_file)

BackgroundVal = np.load(Background_Valfile )
silsVal = np.load(BWShaft_Valfile )
paramsVal = np.load(parameters_Valfile )
# print(np.min(params[:,4]))

#  ------------------------------------------------------------------

ratio = 0.05  # 70%training 30%validation
split = int(len(Background)*ratio)
testlen = 100


train_im = Background[split:]  # 90% training
train_sil = sils[split:]
train_param = params[split:]
number_train_im = np.shape(train_im)[0]
print('we have {} images for the training'.format(number_train_im))


test_im = Background[:split]  # remaining ratio for validation
test_sil = sils[:split]
test_param  = params[:split]
number_test_im = np.shape(test_im)[0]
print('we have {} images for the test '.format(number_test_im))

val_im  = BackgroundVal[100:200]
val_sil  = silsVal[100:200]
val_param = paramsVal[100:200]



#  ------------------------------------------------------------------

normalize = Normalize(mean=[0.5], std=[0.5])
transforms = Compose([ToTensor(),  normalize])
train_dataset = CubeDataset(train_im, train_sil, train_param, transforms)
test_dataset = CubeDataset(test_im, test_sil, test_param, transforms)
val_dataset = CubeDataset(val_im, val_sil, val_param, transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

#
# for image, sil, param in train_dataloader:
#
# #plot silhouette
#     print(image.size(), sil.size(), param.size()) #torch.Size([batch, 3, 512, 512]) torch.Size([batch, 6])
#     im = 0
#     print(param[im])  # parameter in form tensor([2.5508, 0.0000, 0.0000, 0.0000, 0.0000, 5.0000])
#
#     image2show = image[im]  # indexing random  one image
#     print(image2show.size()) #torch.Size([3, 512, 512])
#     plt.imshow((image2show * 0.5 + 0.5).numpy().transpose(1, 2, 0))
#     plt.show()
#
#     image2show = sil[im]  # indexing random  one image
#     print(image2show.size())  # torch.Size([3, 512, 512])
#     image2show = image2show.numpy()
#     plt.imshow(image2show, cmap='gray')
#     plt.show()
#
#     break  # break here just to show 1 batch of data
#
#
# for image, sil, param in test_dataloader:
#
#     nim = image.size()[0]
#     for i in range(0,nim):
#         print(image.size(), sil.size(), param.size()) #torch.Size([batch, 3, 512, 512]) torch.Size([batch, 6])
#         im = i
#         print(param[im])  # parameter in form tensor([2.5508, 0.0000, 0.0000, 0.0000, 0.0000, 5.0000])
#
#
#         image2show = image[im]  # indexing random  one image
#         print(image2show.size()) #torch.Size([3, 512, 512])
#         plt.imshow((image2show * 0.5 + 0.5).numpy().transpose(1, 2, 0))
#         plt.show()
#
#         image2show = sil[im]  # indexing random  one image
#         print(image2show.size())  # torch.Size([3, 512, 512])
#         image2show = image2show.numpy()
#         plt.imshow(image2show, cmap='gray')
#         plt.show()


#  ------------------------------------------------------------------
# Setup the model

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '3D_objects')







noise = 0.0
parser = argparse.ArgumentParser()
parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, '{}.obj'.format(obj_name)))
parser.add_argument('-or', '--filename_output', type=str,default=os.path.join(data_dir, 'example5_resultR_render_1.gif'))
parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()

#camera setting and renderer are part of the model, (model.renderer to reach the renderer function)
model = Myresnet50(filename_obj=args.filename_obj)
# model = Myresnet50(filename_obj=args.filename_obj, cifar = False, modelName='FinalModel_train_15111render_121epochs_testlossdivision')
# model = Myresnet50(filename_obj=args.filename_obj, cifar = False, modelName='211119_100epochtest2_FinalModel_train_Shaft_444_images3_2batchs_101epochs_Noise0.0_100epochtest2_RenderRegrSav')
model = Myresnet50(filename_obj=args.filename_obj, cifar = False, modelName='151119_test_FinalModel_train_Shaft_444_images3_2batchs_100epochs_Noise0.0_test_RenderRegrSave') #good reg result
# model = Myresnet50(filename_obj=args.filename_obj, cifar = False, modelName='FinalModel_train_15111regression_100epochs_test')
# 151119_test_FinalModel_train_Shaft_444_images3_2batchs_100epochs_Noise0.0_test_RenderRegrSave
# 211119_100epochtest2_FinalModel_train_Shaft_444_images3_2batchs_20epochs_Noise0.0_100epochtest2_RenderRegr

# 211119_100epochtest2_FinalModel_train_Shaft_444_images3_2batchs_101epochs_Noise0.0_100epochtest2_RenderRegrSav
model.to(device)

model.train(True)
bool_first = True
lr = 0.0001

criterion = nn.BCELoss()  #nn.BCELoss()   #nn.CrossEntropyLoss()  define the loss (MSE, Crossentropy, Binarycrossentropy)
#
#  ------------------------------------------------------------------
#call renderer

# train_renderV3(model, train_dataloader, test_dataloader,
#                                         n_epochs, criterion,
#                                         date4File, ShaftSetName, batch_size, fileExtension, device, obj_name, noise, number_train_im)

#call regression
train_regV3(model, train_dataloader, test_dataloader,
                                        n_epochs, criterion,
                                        date4File, ShaftSetName, batch_size, fileExtension, device, obj_name, noise, number_train_im, val_dataloader)

#  ------------------------------------------------------------------

#  ------------------------------------------------------------------
