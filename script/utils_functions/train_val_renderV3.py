
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
                    default=os.path.join(sil_dir, 'ResultSilhouette_{}.gif'.format('test1')))
parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()

def sigmoid(x, a, b, c, d):
    """ General sigmoid function
    a adjusts amplitude
    b adjusts y offset
    c adjusts x offset
    d adjusts slope """
    y = ((a-b) / (1 + np.exp(x-(c/2))**d)) + b
    return y




def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()

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

def train_renderV3(model, train_dataloader, test_dataloader,
                 n_epochs, loss_function,
                 date4File, cubeSetName, batch_size, fileExtension, device, obj_name, noise, number_train_im):
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
    epochsValLoss = open(
        "./results/epochsValLoss_{}_{}_RenderRegr_{}.txt".format(date4File,str(n_epochs), fileExtension), "w+")
    TestParamLoss = open(
        "./results/TestParamLoss_{}_{}_RenderRegr_{}.txt".format(date4File,str(n_epochs), fileExtension), "w+")

    x = np.arange(n_epochs)
    y = sigmoid(x, 1, 0, n_epochs/2, 0.2)

    steps_losses = []  # contains the loss after each steps
    steps_alpha_loss = []
    steps_beta_loss = []
    steps_gamma_loss = []
    steps_x_loss = []
    steps_y_loss = []
    steps_z_loss = []

    plt.plot(x, y)
    plt.show()


    for epoch in range(n_epochs):

        ## Training phase
        model.train()
        print('train phase epoch {}/{}'.format(epoch, n_epochs))

        alpha = y[epoch] #proportion of the regression part decrease with negative sigmoid

        print('alpha is {}'.format(alpha))

        t = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))
        for image, silhouette, parameter in t:
            image = image.to(device)
            parameter = parameter.to(device)
            params = model(image)  # should be size [batchsize, 6]


            # print(params)
            numbOfImage = image.size()[0]
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            optimizer.zero_grad()

            for i in range(0,numbOfImage):



                model.t = params[i, 3:6]
                R = params[i, 0:3]
                model.R = R2Rmat(R)
                current_sil = model.renderer(model.vertices, model.faces,  R=model.R, t=model.t, mode='silhouettes').squeeze()
                current_sil =  current_sil[0:1024, 0:1280]
                current_GT_sil = (silhouette[i]/255).type(torch.FloatTensor).to(device)

                if (i == 0):
                    loss = (nn.BCELoss()(current_sil, current_GT_sil).to(device))*(1 - alpha) + (nn.MSELoss()(params[i], parameter[i]).to(device))*(alpha)
                else:
                    loss += (nn.BCELoss()(current_sil, current_GT_sil).to(device)) * (1 - alpha) + (nn.MSELoss()(params[i], parameter[i]).to(device)) *(alpha)

                # if (model.t[2] > 0.0317 and model.t[2] < 0.1 and torch.abs(model.t[0]) < 0.06 and torch.abs(model.t[1]) < 0.06):
                #
                #     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
                #     optimizer.zero_grad()
                #     if (i == 0):
                #         loss  =  nn.BCELoss()(current_sil, current_GT_sil).to(device)
                #     else:
                #         loss = loss + nn.BCELoss()(current_sil, current_GT_sil).to(device)
                #     print('render')
                #     renderCount += 1
                # else:
                #     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
                #     optimizer.zero_grad()
                #     if (i == 0):
                #         loss = nn.MSELoss()(params[i, 3:6], parameter[i, 3:6]).to(device)
                #         # loss = nn.MSELoss()(params[i], parameter[i]).to(device)
                #     else:
                #         loss = loss + nn.MSELoss()(params[i, 3:6], parameter[i, 3:6]).to(device)
                #         # loss = loss + nn.MSELoss()(params[i], parameter[i]).to(device)
                #     print('regression')
                #     regressionCount += 1

            loss = loss/numbOfImage
            loss.backward()
            optimizer.step()

            current_step_loss.append(loss.detach().cpu().numpy())  # contain only this epoch loss, will be reset after each epoch
            count = count + 1

        epochValloss = np.mean(current_step_loss)
        epochsValLoss.write('step: {}/{} current step loss: {:.4f},   \r\n'.format(epoch, n_epochs, epochValloss))
        print('loss of epoch {} is {}'.format(epoch, epochValloss))
        current_step_loss = [] #reset value
        Epoch_Val_losses.append(epochValloss)  # most significant value to store

        # print(epochValloss)

        print('render: {}, regression: {}'.format(renderCount,regressionCount))
        regressionCount = 0
        renderCount = 0

        count = 0
        testcount = 0
        model.eval()

        t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))
        for image, silhouette, parameter in t:

            Test_Step_loss = []
            numbOfImage = image.size()[0]

            image = image.to(device)
            parameter = parameter.to(device)
            params = model(image)  # should be size [batchsize, 6]
            # print(np.shape(params))




            for i in range(0, numbOfImage):
                print('image tested: {}'.format(testcount))
                print('estated {}'.format(params[i]))
                print('Ground Truth {}'.format(parameter[i]))
                if (i == 0):
                    loss = nn.MSELoss()(params[i], parameter[i]).to(device)
                else:
                    loss = loss + nn.MSELoss()(params[i], parameter[i]).to(device)

                # one value each for the step, compute mse loss for all parameters separately
                alpha_loss = nn.MSELoss()(params[:, 0], parameter[:, 0])
                beta_loss = nn.MSELoss()(params[:, 1], parameter[:, 1])
                gamma_loss = nn.MSELoss()(params[:, 2], parameter[:, 2])
                x_loss = nn.MSELoss()(params[:, 3], parameter[:, 3])
                y_loss = nn.MSELoss()(params[:, 4], parameter[:, 4])
                z_loss = nn.MSELoss()(params[:, 5], parameter[:, 5])

                steps_losses.append(loss.item())  # only one loss value is add each step
                steps_alpha_loss.append(alpha_loss.item())
                steps_beta_loss.append(beta_loss.item())
                steps_gamma_loss.append(gamma_loss.item())
                steps_x_loss.append(x_loss.item())
                steps_y_loss.append(y_loss.item())
                steps_z_loss.append(z_loss.item())

                TestParamLoss.write(
                'test image: {} current step loss: {:.4f}, angle loss: {:.4f} {:.4f} {:.4f} translation loss: {:.4f} {:.4f} {:.4f}\r\n'
                .format(testcount, len(loop), loss, alpha_loss, beta_loss, gamma_loss, x_loss, y_loss, z_loss))

            if epoch_count == n_epochs:
                model.t = params[i, 3:6]
                R = params[i, 0:3]
                model.R = R2Rmat(R)  # angle from resnet are in radian

                current_sil = model.renderer(model.vertices, model.faces, R=model.R, t=model.t,
                                             mode='silhouettes').squeeze()
                current_sil = current_sil[0:1024, 0:1280]
                sil2plot = np.squeeze((current_sil.detach().cpu().numpy() * 255)).astype(np.uint8)
                current_GT_sil = (silhouette[i] / 255).type(torch.FloatTensor).to(device)

                fig = plt.figure()
                fig.add_subplot(2, 1, 1)
                plt.imshow(sil2plot, cmap='gray')

                fig.add_subplot(2, 1, 2)
                plt.imshow(silhouette[i], cmap='gray')
                plt.savefig('results/imageRender_{}.png'.format(testcount), bbox_inches='tight',
                            pad_inches=0.05)
                plt.show()

            #MSE loss
            current_step_Test_loss.append(loss.detach().cpu().numpy())
            testcount = testcount + 1

        epochTestloss = np.mean(current_step_Test_loss)
        current_step_Test_loss = [] #reset the loss value
        Epoch_Test_losses.append(epochTestloss)  # most significant value to store
        epoch_count = epoch_count+1



    fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1 = plt.subplot(2, 1, 1)
    ax1.plot(Epoch_Val_losses)
    ax1.set_ylabel('training render BCE loss')
    ax1.set_xlabel('epoch')
    ax1.set_xlim([0, n_epochs])
    ax1.set_ylim([0, 4])
    # ax1.set_yscale('log')


    # ax2 = plt.subplot(2, 1, 2)
    ax2.plot(Epoch_Test_losses)
    ax2.set_ylabel('test render MSE loss')
    ax2.set_xlabel('epoch')
    ax2.set_xlim([0, n_epochs])
    ax2.set_ylim([0, 0.1])
    # ax2.set_yscale('log')


    plt.savefig('results/training_epochs_rend_results_{}.png'.format(fileExtension), bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()

    fig, (a,b,g,x,y,z) = plt.subplots(6, 1)
    a.plot(alpha_loss)
    a.set_ylabel('test alpha loss')

    b.plot(beta_loss)
    b.set_ylabel('test beta loss')

    g.plot(gamma_loss)
    g.set_ylabel('test gamma loss')

    x.plot(x_loss)
    x.set_ylabel('test x loss')

    y.plot(y_loss)
    y.set_ylabel('test y loss')

    z.plot(x_loss)
    z.set_ylabel('test z loss')
    z.set_xlabel('epoch')



    plt.savefig('results/test_epochs_rend_Rt_Loss_{}.png'.format(fileExtension), bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()


    epochsValLoss.close()