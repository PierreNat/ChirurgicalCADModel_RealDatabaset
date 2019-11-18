
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

def train_renderV2(model, train_dataloader, test_dataloader,
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
            # print(parameter )
            params = model(image)  # should be size [batchsize, 6]
            # print(params)
            numbOfImage = image.size()[0]
            # loss = nn.MSELoss()(params, parameter).to(device)

            for i in range(0,numbOfImage):
                #create and store silhouette
                model.t = params[i, 3:6]
                # model.t[0] = 0
                # model.t[1] = 0
                # model.t[2] = 0.08


                # print(model.t)
                R = params[i, 0:3]
                # R[0] = 0
                # R[1] = 0
                # R[2] = 0
                # print(R)
                model.R = R2Rmat(R)  # angle from resnet are in radian, function controlled
                # print(model.R)


                current_sil = model.renderer(model.vertices, model.faces,  R=model.R, t=model.t, mode='silhouettes').squeeze()
                current_sil =  current_sil[0:1024, 0:1280]

                # print(current_sil.size())
                current_GT_sil = (silhouette[i]/255).type(torch.FloatTensor).to(device)

                sil2plot = np.squeeze((current_sil.detach().cpu().numpy()* 255)).astype(np.uint8)

                if count%50 == 0:
                    # fig = plt.figure()
                    # fig.add_subplot(2, 1, 1)
                    # plt.imshow(sil2plot, cmap='gray')
                    #
                    # fig.add_subplot(2, 1, 2)
                    # plt.imshow(silhouette[i], cmap='gray')
                    # plt.show()

                    print('renderer value {}'.format(params[i]))
                    print('ground truth value {}'.format(parameter[i]))


                #regression test to see if the training is done correctly
                optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
                optimizer.zero_grad()
                if (i == 0):
                    # loss = nn.MSELoss()(params[i, 3:6], parameter[i, 3:6]).to(device)
                    loss = nn.MSELoss()(params[i], parameter[i]).to(device)
                else:
                    # loss = loss + nn.MSELoss()(params[i, 3:6], parameter[i, 3:6]).to(device)
                    loss = loss + nn.MSELoss()(params[i], parameter[i]).to(device)

                # if (model.t[2] > 0 and model.t[2] < 0.1 and torch.abs(model.t[0]) < 0.04 and torch.abs(model.t[1]) < 0.04):
                # # if (epoch > 0):
                #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                #     optimizer.zero_grad()
                #     if (i == 0):
                #         loss  =  nn.BCELoss()(current_sil, current_GT_sil).to(device)
                #     else:
                #         loss = loss + nn.BCELoss()(current_sil, current_GT_sil).to(device)
                #     print('render')
                #     renderCount += 1
                # else:
                #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                #     optimizer.zero_grad()
                #     if (i == 0):
                #         # loss = nn.MSELoss()(params[i, 3:6], parameter[i, 3:6]).to(device)
                #         loss = nn.MSELoss()(params[i], parameter[i]).to(device)
                #     else:
                #         # loss = loss + nn.MSELoss()(params[i, 3:6], parameter[i, 3:6]).to(device)
                #         loss = loss + nn.MSELoss()(params[i], parameter[i]).to(device)
                #     print('regression')
                #     regressionCount += 1

            loss.backward()
            optimizer.step()
            print(loss)
            Step_Val_losses.append(loss.detach().cpu().numpy())  # contain all step value for all epoch
            current_step_loss.append(loss.detach().cpu().numpy())  # contain only this epoch loss, will be reset after each epoch
            count = count + 1

        epochValloss = np.mean(current_step_loss)
        current_step_loss = []
        Epoch_Val_losses.append(epochValloss)  # most significant value to store
        # print(epochValloss)

        print(Epoch_Val_losses)
        # print(renderCount, regressionCount)

        renderbar.append(renderCount)
        regressionbar.append(regressionCount)
        renderCount = 0
        regressionCount = 0

        count = 0

        # allstepvalloss.append(Step_Val_losses)
        # plt.plot(Step_Val_losses)
        # plt.ylabel('loss')
        # plt.xlabel('step')
        # plt.ylim(0, 2)
        # plt.show()
    plt.plot(Step_Val_losses)
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.ylim(0, 2)
    plt.show()

        # torch.save(model.state_dict(),
        #            'models/{}epoch_{}_TempModel_train_{}_{}batchs_{}epochs_Noise{}_{}_RenderRegr.pth'.format(epoch, date4File,
        #                                                                                               cubeSetName,
        #                                                                                               str(batch_size),
        #                                                                                               str(n_epochs),
        #                                                                                               noise * 100,
        #                                                                                               fileExtension))
        # test phase phase
    #     print('test phase epoch epoch {}/{}'.format(epoch, n_epochs))
    #     model.eval()
    #
    #     t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))
    #     for image, silhouette, parameter in t:
    #
    #         Test_Step_loss = []
    #         numbOfImage = image.size()[0]
    #         print(numbOfImage)
    #
    #         image = image.to(device)
    #         parameter = parameter.to(device)
    #         params = model(image)  # should be size [batchsize, 6]
    #         # print(np.shape(params))
    #
    #         for i in range(0,numbOfImage):
    #             #create and store silhouette
    #             model.t = params[i, 3:6]
    #             R = params[i, 0:3]
    #             model.R = R2Rmat(R)  # angle from resnet are in radian
    #
    #             current_sil = model.renderer(model.vertices, model.faces, R=model.R, t=model.t, mode='silhouettes').squeeze()
    #             current_sil =  current_sil[0:1024, 0:1280]
    #             current_GT_sil = (silhouette[i]/255).type(torch.FloatTensor).to(device)
    #
    #             if count%10 == 0:
    #                 # fig = plt.figure()
    #                 # fig.add_subplot(2, 1, 1)
    #                 # plt.imshow(sil2plot, cmap='gray')
    #                 #
    #                 # fig.add_subplot(2, 1, 2)
    #                 # plt.imshow(silhouette[i], cmap='gray')
    #                 print('renderer value {}'.format(params[i]))
    #                 print('ground truth value {}'.format(parameter[i]))
    #                 # plt.show()
    #
    #             # # renderer
    #             # if (i == 0):
    #             #     loss  =  nn.BCELoss()(current_sil, current_GT_sil).to(device)
    #             # else:
    #             #     loss = loss + nn.BCELoss()(current_sil, current_GT_sil).to(device) #sum of all loss of each step
    #
    #             # regression
    #
    #             if (i == 0):
    #                 # loss = nn.MSELoss()(params[i, 3:6], parameter[i, 3:6]).to(device)
    #                 loss = nn.MSELoss()(params[i], parameter[i]).to(device)
    #             else:
    #                 # loss = loss + nn.MSELoss()(params[i, 3:6], parameter[i, 3:6]).to(device)
    #                 loss = loss + nn.MSELoss()(params[i], parameter[i]).to(device)
    #
    #         count = count + 1
    #
    #
    #         Test_Step_loss.append(loss.detach().cpu().numpy()) #all step loss of the epoch
    #
    #         if (epoch == n_epochs - 1):  # if we are at the last epoch, save param to plot result
    #
    #             LastEpochTestCPparam.extend(params.detach().cpu().numpy())
    #             LastEpochTestGTparam.extend(parameter.detach().cpu().numpy())
    #
    #         Test_losses.append(loss.detach().cpu().numpy())
    #         current_step_Test_loss.append(loss.detach().cpu().numpy())
    #         testcount = testcount + 1
    #
    #     epochTestloss = np.mean(current_step_Test_loss) #mean of all step loss makes the epoch test loss
    #     current_step_Test_loss = []
    #     Epoch_Test_losses.append(epochTestloss)  # most significant value to store
    #     print('test loss is {}'.format(epochTestloss))
    #
    # plt.plot(epochTestloss)
    # plt.ylabel(' test loss')
    # plt.xlabel('step')
    # # plt.ylim(0, 2)
    # plt.show()
#
# # ----------- plot some result from the last epoch computation ------------------------
#
#         # print(np.shape(LastEpochTestCPparam)[0])
#     nim = 5
#     for i in range(0, nim):
#         print('saving image to show')
#         pickim = int(uniform(0, np.shape(LastEpochTestCPparam)[0] - 1))
#         # print(pickim)
#
#         model.t = torch.from_numpy(LastEpochTestCPparam[pickim][3:6]).to(device)
#         R = torch.from_numpy(LastEpochTestCPparam[pickim][0:3]).to(device)
#         model.R = R2Rmat(R)  # angle from resnet are in radia
#         imgCP, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures), R=model.R, t=model.t)
#
#         model.t = torch.from_numpy(LastEpochTestGTparam[pickim][3:6]).to(device)
#         R = torch.from_numpy(LastEpochTestGTparam[pickim][0:3]).to(device)
#         model.R = R2Rmat(R)  # angle from resnet are in radia
#         imgGT, _, _ = model.renderer(model.vertices, model.faces, torch.tanh(model.textures), R=model.R, t=model.t)
#
#         imgCP = imgCP.squeeze()  # float32 from 0-1
#         imgCP = imgCP.detach().cpu().numpy().transpose((1, 2, 0))
#         imgCP = (imgCP * 255).astype(np.uint8)  # cast from float32 255.0 to 255 uint8
#         imgGT = imgGT.squeeze()  # float32 from 0-1
#         imgGT = imgGT.detach().cpu().numpy().transpose((1, 2, 0))
#         imgGT = (imgGT * 255).astype(np.uint8)  # cast from float32 255.0 to 255 uint8
#         Im2ShowGT.append(imgCP)
#         Im2ShowGCP.append(imgGT)
#
#         a = plt.subplot(2, nim, i + 1)
#         plt.imshow(imgGT)
#         a.set_title('GT {}'.format(i))
#         plt.xticks([0, 512])
#         plt.yticks([])
#         a = plt.subplot(2, nim, i + 1 + nim)
#         plt.imshow(imgCP)
#         a.set_title('Rdr {}'.format(i))
#         plt.xticks([0, 512])
#         plt.yticks([])
#
#     plt.savefig('results/image_render_{}batch_{}_{}.pdf'.format(batch_size, n_epochs, fileExtension))
# #-----------plot and save section ------------------------------------------------------------------------------------
#
#     fig, (p1, p2, p3, p4) = plt.subplots(4, figsize=(15, 15))  # largeur hauteur
#
#
#     moving_aves = RolAv(Step_Val_losses, window=40)
#     ind = np.arange(n_epochs)  # index
#
#     p1.plot(np.arange(np.shape(moving_aves)[0]), moving_aves, label="step Loss rolling average")
#     p1.set(ylabel='BCE Step Loss')
#     p1.set_yscale('log')
#     p1.set(xlabel='Steps')
#     p1.set_ylim([0, 10])
#     p1.legend()  # Place a legend to the right of this smaller subplot.
#
#     # subplot 2
#     p2.plot(np.arange(n_epochs), Epoch_Val_losses, label="Render epoch Loss")
#     p2.set(ylabel=' Mean of BCE training step loss')
#     p2.set(xlabel='Epochs')
#     p2.set_ylim([0, 5])
#     p2.set_xticks(ind)
#     p2.legend()
#
#     # subplot 3
#
#     width = 0.35
#     p3.bar(ind, renderbar, width, color='#d62728', label="render")
#     height_cumulative = renderbar
#     p3.bar(ind, regressionbar, width, bottom=height_cumulative, label="regression")
#     p3.set(ylabel='render/regression call')
#     p3.set(xlabel='Epochs')
#     p3.set_ylim([0, numbOfImageDataset])
#     p3.set_xticks(ind)
#     p3.legend()
#
#     # subplot 4
#     p4.plot(np.arange(n_epochs), Epoch_Test_losses, label="Render Test Loss")
#     p4.set(ylabel='Mean of BCE test step loss')
#     p4.set(xlabel='Epochs')
#     p4.set_ylim([0, 5])
#     p4.legend()
#
#
#     plt.show()
#
#     fig.savefig('results/render_{}batch_{}_{}.pdf'.format(batch_size, n_epochs, fileExtension))
#
#
#     matplotlib2tikz.save("results/render_{}batch_{}_{}.tex".format(batch_size, n_epochs, fileExtension))