
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


# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)
time = datetime.now()
today = '{}{}-{}h{}'.format(time.day, time.month, time.hour, time.minute)
print("Today's date:", today)

##### PARAMETERS GO HERE ###########
batch_size = 2
vallen = 100
n_epochs = 100
lr = 0.00001
useofFK = False #use of the noisy ground truth as mlp layer during the training
validation = False #not implemented
useOwnPretrainedModel = True #continue to train a existing trained network
if useOwnPretrainedModel:
    modelName='FinalModel_train_251119render_100epochs_newshafttest'
else:
    modelName = '/'

date4File = today #mmddyy
obj_name = 'LongShaft2'#'shaftshortOnly'
comment = 'train one image'
traintype = 'render' #'regression' or 'render'
ResnetOutput = 't' #Rt #define if the resnet gives 3 (only translation) or 6 outputs (Rotation and translation)



file_name_extension = '444_images3' #'806_images3'#'444_images3' #'444_images3'  # choose the corresponding database to use
file_name_extension_validation = '693_images2'  # choose the corresponding database to use
ShaftSetName = 'Shaft_{}'.format(file_name_extension) #used to describe the document name
SettingString = 'training with {} epochs, use of fK: {}, use of own model: {}, model: {} \r\n object: {}, training type: {}, \r\n resnetOutput: {},  training dataset used: {}, comment: {}'.format(n_epochs,
                                                                                                                                             useofFK,
                                                                                                                                              useOwnPretrainedModel,
                                                                                                                                              modelName,
                                                                                                                                              obj_name,
                                                                                                                                             traintype,
                                                                                                                                             ResnetOutput,
                                                                                                                                             file_name_extension,
                                                                                                                                                comment,)
print(SettingString)

Background_file = 'Npydatabase/endoscIm_{}.npy'.format(file_name_extension)
RGBshaft_file = 'Npydatabase/RGBShaft_{}.npy'.format(file_name_extension)
BWShaft_file = 'Npydatabase/BWShaft_{}.npy'.format(file_name_extension)
parameters_file = 'Npydatabase/params_{}.npy'.format(file_name_extension)

Background_Valfile = 'Npydatabase/endoscIm_{}.npy'.format(file_name_extension_validation)
RGBshaft_Valfile = 'Npydatabase/RGBShaft_{}.npy'.format(file_name_extension_validation)
BWShaft_Valfile = 'Npydatabase/BWShaft_{}.npy'.format(file_name_extension_validation)
parameters_Valfile = 'Npydatabase/params_{}.npy'.format(file_name_extension_validation)

fileExtension = '{}{}_{}epochs_{}'.format(date4File,traintype, n_epochs,comment) #string to ad at the end of the file

Background = np.load(Background_file)
sils = np.load(BWShaft_file)
params = np.load(parameters_file)

BackgroundVal = np.load(Background_Valfile )
silsVal = np.load(BWShaft_Valfile )
paramsVal = np.load(parameters_Valfile )
# print(np.min(params[:,4]))

#  ------------------------------------------------------------------

ratio = 0.08  # 70%training 30%validation
split = int(len(Background)*ratio)
testlen = 100


train_im = Background[0:1]  # 90% training
train_sil = sils[0:1]
train_param = params[0:1]
number_train_im = np.shape(train_im)[0]
print('we have {} images for the training'.format(number_train_im))


test_im = Background[0:1]  # remaining ratio for validation
test_sil = sils[0:1]
test_param  = params[0:1]
number_test_im = np.shape(test_im)[0]
print('we have {} images for the test '.format(number_test_im))

val_im  = BackgroundVal[:vallen]
val_sil  = silsVal[:vallen]
val_param = paramsVal[:vallen]



#  ------------------------------------------------------------------

normalize = Normalize(mean=[0.5], std=[0.5])
transforms = Compose([ToTensor(),  normalize])
train_dataset = CubeDataset(train_im, train_sil, train_param, transforms)
test_dataset = CubeDataset(test_im, test_sil, test_param, transforms)
val_dataset = CubeDataset(val_im, val_sil, val_param, transforms)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)


for image, sil, param in train_dataloader:

#plot silhouette
    print(image.size(), sil.size(), param.size()) #torch.Size([batch, 3, 512, 512]) torch.Size([batch, 6])
    im = 0
    print(param[im])  # parameter in form tensor([2.5508, 0.0000, 0.0000, 0.0000, 0.0000, 5.0000])

    image2show = image[im]  # indexing random  one image
    print(image2show.size()) #torch.Size([3, 512, 512])
    plt.imshow((image2show * 0.5 + 0.5).numpy().transpose(1, 2, 0))
    plt.show()

    image2show = sil[im]  # indexing random  one image
    print(image2show.size())  # torch.Size([3, 512, 512])
    image2show = image2show.numpy()
    plt.imshow(image2show, cmap='gray')
    plt.show()

    break  # break here just to show 1 batch of data
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

# shall we continue training an existing model or start from scratch?
if useOwnPretrainedModel:
    if ResnetOutput == 't':  # resnet predict only translation parameter
        print('own model used is t')
        model = Myresnet50_t(filename_obj=args.filename_obj, cifar = False, modelName=modelName)

    if ResnetOutput == 'Rt':  # resnet predict rotation and translation
        print('own model used is Rt')
        model = Myresnet50(filename_obj=args.filename_obj, cifar = False, modelName=modelName)
else:
    if ResnetOutput == 't': #resnet predict only translation parameter
        print('train model used is t')
        model = Myresnet50_t(filename_obj=args.filename_obj)

    if ResnetOutput == 'Rt': #resnet predict rotation and translation
        print('train model used is Rt')
        model = Myresnet50(filename_obj=args.filename_obj)



#camera setting and renderer are part of the model, (model.renderer to reach the renderer function)
# model = Myresnet50(filename_obj=args.filename_obj)
# model = Myresnet50(filename_obj=args.filename_obj, cifar = False, modelName='211119_100epochtest2_FinalModel_train_Shaft_444_images3_2batchs_20epochs_Noise0.0_100epochtest2_RenderRegr')
# model = Myresnet50(filename_obj=args.filename_obj, cifar = False, modelName='211119_100epochtest2_FinalModel_train_Shaft_444_images3_2batchs_101epochs_Noise0.0_100epochtest2_RenderRegrSav')
# model = Myresnet50(filename_obj=args.filename_obj, cifar = False, modelName='151119_test_FinalModel_train_Shaft_444_images3_2batchs_100epochs_Noise0.0_test_RenderRegrSave') #good reg result
# model = Myresnet50(filename_obj=args.filename_obj, cifar = False, modelName='151119_test_FinalModel_train_Shaft_444_images3_2batchs_60epochs_Noise0.0_test_RenderRegrSave')
# 151119_test_FinalModel_train_Shaft_444_images3_2batchs_100epochs_Noise0.0_test_RenderRegrSave
# 211119_100epochtest2_FinalModel_train_Shaft_444_images3_2batchs_20epochs_Noise0.0_100epochtest2_RenderRegr

# 211119_100epochtest2_FinalModel_train_Shaft_444_images3_2batchs_101epochs_Noise0.0_100epochtest2_RenderRegrSav
model.to(device)

model.train(True)
bool_first = True


#  ------------------------------------------------------------------
#call training

training(model, train_dataloader, test_dataloader, val_dataloader, n_epochs, fileExtension, device, traintype, lr, validation, number_test_im, useofFK, ResnetOutput, SettingString, useOwnPretrainedModel)

#call regression
# train_regV3(model, train_dataloader, test_dataloader,
#                                         n_epochs, criterion,
#                                         date4File, ShaftSetName, batch_size, fileExtension, device, obj_name, noise, number_train_im, val_dataloader)

#  ------------------------------------------------------------------

# torch.save(model.state_dict(), 'models/{}_FinalModel_train_{}_{}batchs_{}epochs.pth'.format(date4File, traintype, str(batch_size), str(n_epochs)))
# print('parameters saved')

#  ------------------------------------------------------------------
