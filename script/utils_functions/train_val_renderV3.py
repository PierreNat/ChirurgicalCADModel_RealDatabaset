
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import  pandas as pd
import matplotlib.pyplot as plt
from utils_functions.camera_settings import camera_setttings
import neural_renderer as nr
from utils_functions.R2Rmat import R2Rmat
from utils_functions.camera_settings import BuildTransformationMatrix
from numpy.random import uniform
import matplotlib2tikz
# plt.switch_backend('agg')


##### PARAMETERS GO HERE ###########

c_x = 590
c_y = 508

f_x = 1067
f_y = 1067

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


    for epoch in range(n_epochs):

        ## Training phase
        model.train()
        print('train phase epoch {}/{}'.format(epoch, n_epochs))


        t = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))
        for image, silhouette, parameter in t:
            image = image.to(device)
            parameter = parameter.to(device)
            params = model(image)  # should be size [batchsize, 6]


            # print(params)
            numbOfImage = image.size()[0]

            for i in range(0,numbOfImage):

                model.t = params[i, 3:6]
                R = params[i, 0:3]
                model.R = R2Rmat(R)
                current_sil = model.renderer(model.vertices, model.faces,  R=model.R, t=model.t, mode='silhouettes').squeeze()
                current_sil =  current_sil[0:1024, 0:1280]
                current_GT_sil = (silhouette[i]/255).type(torch.FloatTensor).to(device)

                if (model.t[2] > 0.0317 and model.t[2] < 0.1 and torch.abs(model.t[0]) < 0.06 and torch.abs(model.t[1]) < 0.06):

                    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
                    optimizer.zero_grad()
                    if (i == 0):
                        loss  =  nn.BCELoss()(current_sil, current_GT_sil).to(device)
                    else:
                        loss = loss + nn.BCELoss()(current_sil, current_GT_sil).to(device)
                    print('render')
                    renderCount += 1
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
                    optimizer.zero_grad()
                    if (i == 0):
                        # loss = nn.MSELoss()(params[i, 3:6], parameter[i, 3:6]).to(device)
                        loss = nn.MSELoss()(params[i], parameter[i]).to(device)
                    else:
                        # loss = loss + nn.MSELoss()(params[i, 3:6], parameter[i, 3:6]).to(device)
                        loss = loss + nn.MSELoss()(params[i], parameter[i]).to(device)
                    print('regression')
                    regressionCount += 1


            loss.backward()
            optimizer.step()

            print(loss)
            current_step_loss.append(loss.detach().cpu().numpy())  # contain only this epoch loss, will be reset after each epoch
            count = count + 1

        epochValloss = np.mean(current_step_loss)
        current_step_loss = [] #reset value
        Epoch_Val_losses.append(epochValloss)  # most significant value to store
        # print(epochValloss)

        print(Epoch_Val_losses)

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
                plt.savefig('results/image_{}.png'.format(testcount), bbox_inches='tight',
                            pad_inches=0.05)
                plt.show()

            #MSE loss
            current_step_Test_loss.append(loss.detach().cpu().numpy())
            testcount = testcount + 1

        epochTestloss = np.mean(current_step_Test_loss)
        current_step_Test_loss = [] #reset the loss value
        Epoch_Test_losses.append(epochTestloss)  # most significant value to store
        epoch_count = epoch_count+1


    # plt.plot(Epoch_Test_losses)
    # plt.ylabel('loss')
    # plt.xlabel('step')
    # plt.ylim(0, 2)


    a = plt.subplot(2, 1, 1)
    plt.plot(Epoch_Val_losses)
    plt.ylabel('training Render BCE loss')
    plt.xlabel('epoch')
    plt.ylim(0, 1)

    a = plt.subplot(2, 1, 2)
    plt.plot(Epoch_Test_losses)
    plt.ylabel('test Render MSE loss')
    plt.xlabel('epoch')
    plt.ylim(0, 0.1)


    plt.savefig('results/training_epochs_rend_results_{}.png'.format(fileExtension), bbox_inches='tight', pad_inches=0.05)
    plt.show()