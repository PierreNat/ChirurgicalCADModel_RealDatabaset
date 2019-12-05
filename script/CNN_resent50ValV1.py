
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
from utils_functions.Val_V1 import Val_V1
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
vallen = 400 #number of frame in the gif result
start = 0 #start at the xth image of the validation dataset
n_epochs = 100
lr = 0.0001
useofFK = False #use of the noisy ground truth as mlp layer during the training
validation = True #not implemented
useOwnPretrainedModel = True #continue to train a existing trained network

modelName='FinalModel_train_261119render_40epochs_continue existing model2'
# FinalModel_train_261119render_40epochs_continue existing model2


date4File = today #mmddyy
obj_name = 'LongShaft2'#'LongShaft2'#'shaftshortOnly'
comment = 'validation render shift xy 50 '
traintype = ' ' #'regression' or 'render'
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


Background_Valfile = 'Npydatabase/endoscIm_{}.npy'.format(file_name_extension_validation)
RGBshaft_Valfile = 'Npydatabase/RGBShaft_{}.npy'.format(file_name_extension_validation)
BWShaft_Valfile = 'Npydatabase/BWShaft_{}.npy'.format(file_name_extension_validation)
parameters_Valfile = 'Npydatabase/params_{}.npy'.format(file_name_extension_validation)

fileExtension = '{}{}_{}epochs_{}'.format(date4File,traintype, n_epochs,comment) #string to ad at the end of the file


BackgroundVal = np.load(Background_Valfile )
silsVal = np.load(BWShaft_Valfile )
paramsVal = np.load(parameters_Valfile )
# print(np.min(params[:,4]))

#  ------------------------------------------------------------------


val_im  = BackgroundVal[start:start+vallen] #100:200
val_sil  = silsVal[start:start+vallen]
val_param = paramsVal[start:start+vallen]



#  ------------------------------------------------------------------

normalize = Normalize(mean=[0.5], std=[0.5])
transforms = Compose([ToTensor(),  normalize])
val_dataset = CubeDataset(val_im, val_sil, val_param, transforms)

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

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
if validation:
    if ResnetOutput == 't':  # resnet predict only translation parameter
        print('own model used is t')
        model = Myresnet50_t(filename_obj=args.filename_obj, cifar = False, modelName=modelName)
        model.to(device)

    if ResnetOutput == 'Rt':  # resnet predict rotation and translation
        print('own model used is Rt')
        model = Myresnet50(filename_obj=args.filename_obj, cifar = False, modelName=modelName)
        model.to(device)




#camera setting and renderer are part of the model, (model.renderer to reach the renderer function)
# model = Myresnet50(filename_obj=args.filename_obj)
# model = Myresnet50(filename_obj=args.filename_obj, cifar = False, modelName='211119_100epochtest2_FinalModel_train_Shaft_444_images3_2batchs_20epochs_Noise0.0_100epochtest2_RenderRegr')
# model = Myresnet50(filename_obj=args.filename_obj, cifar = False, modelName='211119_100epochtest2_FinalModel_train_Shaft_444_images3_2batchs_101epochs_Noise0.0_100epochtest2_RenderRegrSav')
# model = Myresnet50(filename_obj=args.filename_obj, cifar = False, modelName='151119_test_FinalModel_train_Shaft_444_images3_2batchs_100epochs_Noise0.0_test_RenderRegrSave') #good reg result
# model = Myresnet50(filename_obj=args.filename_obj, cifar = False, modelName='151119_test_FinalModel_train_Shaft_444_images3_2batchs_60epochs_Noise0.0_test_RenderRegrSave')
# 151119_test_FinalModel_train_Shaft_444_images3_2batchs_100epochs_Noise0.0_test_RenderRegrSave
# 211119_100epochtest2_FinalModel_train_Shaft_444_images3_2batchs_20epochs_Noise0.0_100epochtest2_RenderRegr

# 211119_100epochtest2_FinalModel_train_Shaft_444_images3_2batchs_101epochs_Noise0.0_100epochtest2_RenderRegrSav






#  ------------------------------------------------------------------
#call validation
Val_V1(model, val_dataloader, n_epochs, fileExtension, device, traintype, lr,
               validation, useofFK, ResnetOutput, SettingString, useOwnPretrainedModel)

#call regression
# train_regV3(model, train_dataloader, test_dataloader,
#                                         n_epochs, criterion,
#                                         date4File, ShaftSetName, batch_size, fileExtension, device, obj_name, noise, number_train_im, val_dataloader)

#  ------------------------------------------------------------------

# torch.save(model.state_dict(), 'models/{}_FinalModel_train_{}_{}batchs_{}epochs.pth'.format(date4File, traintype, str(batch_size), str(n_epochs)))
# print('parameters saved')

#  ------------------------------------------------------------------
