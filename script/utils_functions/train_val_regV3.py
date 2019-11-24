
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

current_dir = os.path.dirname(os.path.realpath(__file__))
sil_dir = os.path.join(current_dir, 'SilOutput')

parser = argparse.ArgumentParser()
parser.add_argument('-or', '--filename_output', type=str,
                    default=os.path.join(sil_dir, 'ResultrenderSilhouette_{}.gif'.format('0')))
parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()


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

def train_regV3(model, train_dataloader, test_dataloader,
                 n_epochs, loss_function,
                 date4File, cubeSetName, batch_size, fileExtension, device, obj_name, noise, number_train_im, val_dataloader):
    # monitor loss functions as the training progresses

    loop = n_epochs
    Step_Val_losses = []
    current_step_loss = []
    current_step_Test_loss = []
    Test_losses = []
    Epoch_Val_losses = []
    allstepvalloss = []
    Epoch_Test_losses = []
    count = 0
    epoch_count = 1
    testcount = 0
    Im2ShowGT = []
    Im2ShowGCP = []
    LastEpochTestCPparam = []
    LastEpochTestGTparam = []
    numbOfImageDataset = number_train_im
    renderCount = 0
    regressionCount = 0
    renderbar = []
    regressionbar = []
    lr= 0.00001

# training -------------------------------------------------------------------------------------------------------

    # for epoch in range(n_epochs):
    #
    #     ## Training phase
    #     model.train()
    #     print('train phase epoch {}/{}'.format(epoch, n_epochs))
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #
    #     t = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))
    #     for image, silhouette, parameter in t:
    #         image = image.to(device)
    #         parameter = parameter.to(device)
    #         params = model(image)  # should be size [batchsize, 6]
    #         optimizer.zero_grad()
    #
    #         # print(params)
    #         numbOfImage = image.size()[0]
    #         # loss = nn.MSELoss()(params, parameter).to(device)
    #
    #
    #         for i in range(0,numbOfImage):
    #             #create and store silhouette
    #             if (i == 0):
    #                 loss =nn.MSELoss()(params[i], parameter[i]).to(device)
    #             else:
    #                 loss = loss + nn.MSELoss()(params[i], parameter[i]).to(device)
    #
    #
    #         loss.backward()
    #         optimizer.step()
    #
    #         print(loss)
    #         current_step_loss.append(loss.detach().cpu().numpy())  # contain only this epoch loss, will be reset after each epoch
    #         count = count + 1
    #
    #     epochValloss = np.mean(current_step_loss)
    #     current_step_loss = [] #reset value
    #     Epoch_Val_losses.append(epochValloss)  # most significant value to store
    #     # print(epochValloss)
    #
    #     print(Epoch_Val_losses)
    #
    #
    #     #test --------------------------------------------------------------------------------------------------------
    #
    #     count = 0
    #     testcount = 0
    #     model.eval()
    #
    #     t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))
    #     for image, silhouette, parameter in t:
    #
    #         Test_Step_loss = []
    #         numbOfImage = image.size()[0]
    #
    #         image = image.to(device)
    #         parameter = parameter.to(device)
    #         params = model(image)  # should be size [batchsize, 6]
    #         # print(np.shape(params))
    #
    #         for i in range(0, numbOfImage):
    #             print('image tested: {}'.format(testcount))
    #             print('estated {}'.format(params[i]))
    #             print('Ground Truth {}'.format(parameter[i]))
    #             if (i == 0):
    #                 loss = nn.MSELoss()(params[i], parameter[i]).to(device)
    #             else:
    #                 loss = loss + nn.MSELoss()(params[i], parameter[i]).to(device)
    #
    #         if epoch_count == n_epochs:
    #             model.t = params[i, 3:6]
    #             R = params[i, 0:3]
    #             model.R = R2Rmat(R)  # angle from resnet are in radian
    #
    #             current_sil = model.renderer(model.vertices, model.faces, R=model.R, t=model.t,
    #                                          mode='silhouettes').squeeze()
    #             current_sil = current_sil[0:1024, 0:1280]
    #             sil2plot = np.squeeze((current_sil.detach().cpu().numpy() * 255)).astype(np.uint8)
    #             current_GT_sil = (silhouette[i] / 255).type(torch.FloatTensor).to(device)
    #
    #             fig = plt.figure()
    #             fig.add_subplot(2, 1, 1)
    #             plt.imshow(sil2plot, cmap='gray')
    #
    #             fig.add_subplot(2, 1, 2)
    #             plt.imshow(silhouette[i], cmap='gray')
    #             plt.savefig('results/image_{}.png'.format(testcount), bbox_inches='tight',
    #                         pad_inches=0.05)
    #             # plt.show()
    #
    #         current_step_Test_loss.append(loss.detach().cpu().numpy())
    #         testcount = testcount + 1
    #
    #     epochTestloss = np.mean(current_step_Test_loss)
    #     current_step_Test_loss = [] #reset the loss value
    #     Epoch_Test_losses.append(epochTestloss)  # most significant value to store
    #     epoch_count = epoch_count+1
    #
    #
    # # plt.plot(Epoch_Test_losses)
    # # plt.ylabel('loss')
    # # plt.xlabel('step')
    # # plt.ylim(0, 2)
    #
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # # ax1 = plt.subplot(2, 1, 1)
    # ax1.plot(Epoch_Val_losses)
    # ax1.set_ylabel('training loss')
    # ax1.set_xlabel('epoch')
    # ax1.set_xlim([0, n_epochs])
    # ax1.set_ylim([0, 0.4])
    # ax1.set_yscale('log')
    # # ax1.ylim(0, 0.4)
    #
    # # ax2 = plt.subplot(2, 1, 2)
    # ax2.plot(Epoch_Test_losses)
    # ax2.set_ylabel('test loss')
    # ax2.set_xlabel('epoch')
    # ax2.set_xlim([0, n_epochs])
    # ax2.set_ylim([0, 0.1])
    # ax2.set_yscale('log')
    # # ax2.ylim(0, 0.08)
    #
    #
    # plt.savefig('results/training_epochs_results_reg_{}.png'.format(fileExtension), bbox_inches='tight', pad_inches=0.05)
    # # plt.show()

    # validation --------------------------------------------------------------------------------------------------------

    print('validation phase')
    Valcount = 0
    processcount = 0
    step_Val_loss =[]
    Epoch_Val_losses= []
    from PIL import ImageTk, Image, ImageDraw
    epochsValLoss = open(
        "./results/valbothParamShift.txt", "w+")

    t = tqdm(iter(val_dataloader), leave=True, total=len(val_dataloader))
    for image, silhouette, parameter in t:

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
        params = model(image1)  # should be size [batchsize, 6]
        # print(np.shape(params))
        i=0

        # print('image estimated: {}'.format(testcount))
        # print('estimated parameter {}'.format(params[i]))
        # print('Ground Truth {}'.format(parameter[i]))

        epochsValLoss.write('step:{} params:{} \r\n'.format(processcount, params.detach().cpu().numpy()))
        loss = nn.MSELoss()(params[i], parameter[i]).to(device)



        model.t = params[i, 3:6]
        R = params[i, 0:3]
        model.R = R2Rmat(R)  # angle from resnet are in radian


        current_sil = model.renderer(model.vertices, model.faces, R=model.R, t=model.t,
                                     mode='silhouettes').squeeze()
        current_sil = current_sil[0:1024, 0:1280]


        sil2plot = np.squeeze((current_sil.detach().cpu().numpy() * 255)).astype(np.uint8)

        image2show =  np.squeeze((Origimagesave[i].detach().cpu().numpy()))
        image2show = (image2show * 0.5 + 0.5).transpose(1, 2, 0)
        # image2show = np.flip(image2show,1)


        # fig = plt.figure()
        # fig.add_subplot(2, 1, 1)
        # plt.imshow(sil2plot, cmap='gray')
        #
        # fig.add_subplot(2, 1, 2)
        # plt.imshow(image2show)
        # plt.show()


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
