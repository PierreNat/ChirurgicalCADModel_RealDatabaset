
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



    current_step_Train_loss  = []
    Epoch_Train_losses = []


    count = 0
    epoch_count = 1
    renderCount = 0
    regressionCount = 0
    lr= 0.001
    print('lr used is: {}'.format(lr))

    output_result_dir = 'results/render/{}_lr{}'.format(fileExtension,lr)
    mkdir_p(output_result_dir)
    
    epochsTrainLoss = open(
        "{}/epochsTrainLoss_RenderRegr_{}.txt".format(output_result_dir, fileExtension), "w+")
    TestParamLoss = open(
        "{}/TestParamLoss_RenderRegr_{}.txt".format(output_result_dir, fileExtension), "w+")

    x = np.arange(n_epochs)
    y = sigmoid(x, 1, 0, n_epochs/2, 0.2)
    plt.plot(x, y)
    plt.savefig('{}/ReverseSigmoid_{}.png'.format(output_result_dir, fileExtension), bbox_inches='tight', pad_inches=0.05)
    plt.show()


    #usefull for the loss plot of each parameter
    epoch_test_loss = []
    epoch_test_alpha_loss = []
    epoch_test_beta_loss = []
    epoch_test_gamma_loss = []
    epoch_test_x_loss = []
    epoch_test_y_loss = []
    epoch_test_z_loss = []




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
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

            current_step_Train_loss .append(loss.detach().cpu().numpy())  # contain only this epoch loss, will be reset after each epoch
            count = count + 1

        epochTrainloss = np.mean(current_step_Train_loss )
        epochsTrainLoss.write('step: {}/{} current step loss: {:.4f}\r\n'.format(epoch, n_epochs, epochTrainloss))
        print('loss of epoch {} is {}'.format(epoch, epochTrainloss))
        current_step_Train_loss  = [] #reset value
        Epoch_Train_losses.append(epochTrainloss)  # most significant value to store

        # print(epochTrainloss)

        print('render: {}, regression: {}'.format(renderCount,regressionCount))
        regressionCount = 0
        renderCount = 0

        count = 0
        testcount = 0


        model.eval()

        current_step_Test_loss = []
        Epoch_Test_losses = []
        steps_losses = []  # reset the list after each epoch
        steps_alpha_loss = []
        steps_beta_loss = []
        steps_gamma_loss = []
        steps_x_loss = []
        steps_y_loss = []
        steps_z_loss = []

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
                alpha_loss = nn.MSELoss()(params[:, 0], parameter[:, 0]).detach().cpu().numpy()
                beta_loss = nn.MSELoss()(params[:, 1], parameter[:, 1]).detach().cpu().numpy()
                gamma_loss = nn.MSELoss()(params[:, 2], parameter[:, 2]).detach().cpu().numpy()
                x_loss = nn.MSELoss()(params[:, 3], parameter[:, 3]).detach().cpu().numpy()
                y_loss = nn.MSELoss()(params[:, 4], parameter[:, 4]).detach().cpu().numpy()
                z_loss = nn.MSELoss()(params[:, 5], parameter[:, 5]).detach().cpu().numpy()

                steps_losses.append(loss.item())  # only one loss value is add each step
                steps_alpha_loss.append(alpha_loss.item())
                steps_beta_loss.append(beta_loss.item())
                steps_gamma_loss.append(gamma_loss.item())
                steps_x_loss.append(x_loss.item())
                steps_y_loss.append(y_loss.item())
                steps_z_loss.append(z_loss.item())



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

        epoch_test_loss.append(np.mean(steps_losses))
        epoch_test_alpha_loss.append(np.mean(steps_alpha_loss))
        epoch_test_beta_loss.append(np.mean(steps_beta_loss))
        epoch_test_gamma_loss.append(np.mean(steps_gamma_loss))
        epoch_test_x_loss.append(np.mean(steps_x_loss))
        epoch_test_y_loss.append(np.mean(steps_y_loss))
        epoch_test_z_loss.append(np.mean(steps_z_loss))

        TestParamLoss.write(
            'test epoch: {} current step loss: {:.4f}, angle loss: {:.4f} {:.4f} {:.4f} translation loss: {:.4f} {:.4f} {:.4f}\r\n'
                .format(epoch, np.mean(steps_losses),
                        np.mean(steps_alpha_loss),
                        np.mean(steps_beta_loss),
                        np.mean(steps_x_loss),
                        np.mean(steps_x_loss),
                        np.mean(steps_z_loss),
                        np.mean(steps_y_loss)))

        epochTestloss = np.mean(current_step_Test_loss)
        Epoch_Test_losses.append(epochTestloss)  # most significant value to store
        epoch_count = epoch_count+1



    fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1 = plt.subplot(2, 1, 1)
    ax1.plot(Epoch_Train_losses)
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


    plt.savefig('{}/training_epochs_rend_results_{}.png'.format(output_result_dir,fileExtension), bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()

    fig, (a,b,g,x,y,z) = plt.subplots(6, 1)
    a.plot(epoch_test_alpha_loss)
    a.set_xlim([0, n_epochs])
    a.set_ylabel('test alpha loss')

    b.plot(epoch_test_beta_loss)
    b.set_xlim([0, n_epochs])
    b.set_ylabel('test beta loss')

    g.plot(epoch_test_gamma_loss)
    g.set_xlim([0, n_epochs])
    g.set_ylabel('test gamma loss')

    x.plot(epoch_test_x_loss)
    x.set_xlim([0, n_epochs])
    x.set_ylabel('test x loss')

    y.plot(epoch_test_y_loss)
    y.set_xlim([0, n_epochs])
    y.set_ylabel('test y loss')

    z.plot(epoch_test_z_loss)
    z.set_xlim([0, n_epochs])
    z.set_ylabel('test z loss')
    z.set_xlabel('epoch')



    plt.savefig('{}/test_epochs_rend_Rt_Loss_{}.png'.format(output_result_dir,fileExtension), bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()


    epochsTrainLoss.close()
    TestParamLoss .close()