
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import  pandas as pd
import matplotlib.pyplot as plt
from utils_functions.camera_settings import camera_setttings
import neural_renderer as nr
from utils_functions.R2Rmat import R2Rmat
import os
import imageio
import glob
import argparse
from skimage.io import imread, imsave
from utils_functions.camera_settings import BuildTransformationMatrix
from numpy.random import uniform
import matplotlib2tikz
# plt.switch_backend('agg')


##### PARAMETERS GO HERE ###########

c_x = 590
c_y = 508

f_x = 1067
f_y = 1067




def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()

def shiftPixel(image=None, shift=0, axis= 'y'):

    if axis == 'x':

        image = torch.roll(image, shift, 3)
        image[0, :, :, 0:shift] = 0
    if axis == 'y':

        image = torch.roll(image, shift, 2)
        image[0, :, 0:shift,: ] = 0

    return image

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise



def RolAv(list, window = 2):

    mylist = list
    print(mylist)
    N = window
    cumsum, moving_aves = [0], []

    for i, x in enumerate(mylist, 1):
        cumsum.append(cumsum[i - 1] + x)
        if i >= N:
            moving_ave = (cumsum[i] - cumsum[i - N]) / N
            # can do stuff with moving_ave here
            moving_aves.append(moving_ave)

    return moving_aves

def Val_V1(model, val_dataloader, n_epochs, fileExtension, device, traintype, lr, validation, useofFK, ResnetOutput, SettingString, useOwnPretrainedModel):



    output_result_dir = 'results/Validation_{}{}/{}'.format(traintype, ResnetOutput,fileExtension)
    mkdir_p(output_result_dir)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    sil_dir = os.path.join(output_result_dir, 'SilOutput')

    parser = argparse.ArgumentParser()
    parser.add_argument('-or', '--filename_output', type=str,
                        default=os.path.join(sil_dir, 'ValidationGif_{}.gif'.format('LongShaft_2')))
    parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

     # validation --------------------------------------------------------------------------------------------------------

    print('validation phase')
    Valcount = 0
    processcount = 0
    step_Val_loss =[]
    Epoch_Val_losses= []
    model.eval()
    from PIL import ImageTk, Image, ImageDraw
    epochsValLoss = open("./{}/valbothParamShift.txt".format(output_result_dir), "w+")

    t = tqdm(iter(val_dataloader), leave=True, total=len(val_dataloader))
    for image, silhouette, parameter in t:

        i=0
        Test_Step_loss = []
        numbOfImage = image.size()[0]
        # image1 = torch.flip(image,[0, 3]) #flip vertical
        # image = torch.roll(image, 100, 3) #shift down from 100 px
        # image1 = shiftPixel(image, 100, 'y')
        # image1 =   shiftPixel(image , 100, 'x')
        # image1 = torch.flip(image1, [0, 3])
        image1 = image
        Origimagesave = image1
        # Origimagesave = image.to(device)
        image1 = image1.to(device) #torch.Size([1, 3, 1024, 1280])
        parameter = parameter.to(device)

        #parameter estimation through trained model
        if ResnetOutput == 't':  # resnet predict only translation parameter
            print('own model used is t')
            t_params = model(image1)
            model.t = t_params[i]
            R = parameter[i, 0:3]  # give the ground truth parameter for the rotation values
            model.R = R2Rmat(R)
            epochsValLoss.write('step:{} params:{} \r\n'.format(processcount, t_params.detach().cpu().numpy()))
            loss = nn.MSELoss()(params[i], parameter[i]).to(device)


        if ResnetOutput == 'Rt':  # resnet predict rotation and translation
            params = model(image1)
            print('own model used is Rt')
            model.t = params[i, 3:6]
            R = params[i, 0:3]
            model.R = R2Rmat(R)  # angle from resnet are in radian
            epochsValLoss.write('step:{} params:{} \r\n'.format(processcount, params.detach().cpu().numpy()))
            loss = nn.MSELoss()(params[i], parameter[i]).to(device)


        # print(np.shape(params))
        i=0

        # print('image estimated: {}'.format(testcount))
        # print('estimated parameter {}'.format(params[i]))
        # print('Ground Truth {}'.format(parameter[i]))


        current_sil = model.renderer(model.vertices, model.faces, R=model.R, t=model.t,
                                     mode='silhouettes').squeeze()
        current_sil = current_sil[0:1024, 0:1280]


        sil2plot = np.squeeze((current_sil.detach().cpu().numpy() * 255)).astype(np.uint8)

        image2show =  np.squeeze((Origimagesave[i].detach().cpu().numpy()))
        image2show = (image2show * 0.5 + 0.5).transpose(1, 2, 0)
        # image2show = np.flip(image2show,1)


        fig = plt.figure()
        fig.add_subplot(2, 1, 1)
        plt.imshow(sil2plot, cmap='gray')

        fig.add_subplot(2, 1, 2)
        plt.imshow(image2show)
        plt.show()

        #creation of the blender image
        sil3d =  sil2plot[:, :, np.newaxis]
        renderim = np.concatenate((sil3d,sil3d,sil3d), axis=2)
        toolIm = Image.fromarray(np.uint8(renderim))

        # print(type(image2show))
        backgroundIm = Image.fromarray(np.uint8(image2show*255))

        # backgroundIm.show()
        #

        alpha = 0.2
        out = Image.blend(backgroundIm,toolIm,alpha)
        # #make gif
        imsave('/tmp/_tmp_%04d.png' % processcount, np.array(out))
        processcount = processcount + 1
        step_Val_loss.append(loss.detach().cpu().numpy())
        # print(processcount)

    print('making the gif')
    make_gif(args.filename_output)
    
    print(step_Val_loss)
    print(np.mean(step_Val_loss))
    epochsValLoss.close()
    # Valcount = Valcount + 1
    #
    # epochValloss = np.mean(step_Val_loss)
    # Epoch_Val_losses.append(epochValloss)  # most significant value to store
