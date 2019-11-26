
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils_functions.R2Rmat import R2Rmat
import os
import imageio
import glob
import argparse
from skimage.io import imread, imsave
import matplotlib2tikz
plt.switch_backend('agg')



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

def FKBuild(parameters, AngleNoise, Translation_noise):
    # print(parameters.size(), parameters.size()[0])
    # print(parameters)
    # print(parameters[0][3])
    Noisyparameters = parameters

    for i in range(parameters.size()[0]): # the size is the number of item in the batch
        for j in range(parameters.size()[1]): #go trough all extrinsic parameter parameters[i][0 to  5]
            if j <= 2: #angle modification
                # print('modif angle')

                Noisyparameters[i][j] = parameters[i][j] + AngleNoise
            else:
                # print('modif translation')
                Noisyparameters[i][j] = parameters[i][j] + Translation_noise

    Noisyparameters = parameters
    # print(Noisyparameters)
    return Noisyparameters


def training(model, train_dataloader, test_dataloader, val_dataloader, n_epochs, fileExtension, device, traintype, lr, validation, number_test_im, useofFK, ResnetOutput, SettingString ):
    # monitor loss functions as the training progresses


    current_step_Train_loss  = []
    Epoch_Train_losses = []


    count = 0
    epoch_count = 1
    lr= lr
    print('training {}'.format(traintype))
    print('lr used is: {}'.format(lr))

    output_result_dir = 'results/{}{}/{}_lr{}'.format(traintype, ResnetOutput,fileExtension,lr)
    mkdir_p(output_result_dir)


    
    epochsTrainLoss = open(
        "{}/epochsTrainLoss_{}.txt".format(output_result_dir, fileExtension), "w+")
    TestParamLoss = open(
        "{}/TestParamLoss_{}.txt".format(output_result_dir, fileExtension), "w+")
    ExperimentSettings = open(
        "{}/expSettings_{}.txt".format(output_result_dir, fileExtension), "w+")

    ExperimentSettings.write(SettingString)
    ExperimentSettings.close()

    x = np.arange(n_epochs)
    y = sigmoid(x, 1, 0, n_epochs/1.2, 0.1)
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

    Epoch_Test_losses = []


    for epoch in range(n_epochs):

        ## training phase --------------------------------------------------------------------------------------------------------
        model.train()
        print('train phase epoch {}/{}'.format(epoch, n_epochs))

        if useofFK:
            alpha = -1#y[epoch] #proportion of the regression part decrease with negative sigmoid
        else:
            alpha = y[epoch]


        print('alpha is {}'.format(alpha))

        t = tqdm(iter(train_dataloader), leave=True, total=len(train_dataloader))
        for image, silhouette, parameter in t:
            image = image.to(device)

            if useofFK:
                FKparameters = FKBuild(parameter,np.radians(2),0.001) #add noise to the ground truth parameter, degree and cm
                FKparameters = FKparameters.to(device)
                params = model(image, FKparameters, useofFK)  # should be size [batchsize, 6]

            else:
                if ResnetOutput == 'Rt':
                    params = model(image) #call the 6 parameters model
                if ResnetOutput == 't':
                    t_params = model(image) #call the 3 parameters model
                    # print(t_params.size())



            parameter = parameter.to(device)


            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            optimizer.zero_grad()

            numbOfImage = image.size()[0]

            for i in range(0,numbOfImage):
                if ResnetOutput == 'Rt':
                    print('Renset output 6 values')
                    model.t = params[i, 3:6]
                    R = params[i, 0:3]
                    model.R = R2Rmat(R)
                    current_sil = model.renderer(model.vertices, model.faces,  R=model.R, t=model.t, mode='silhouettes').squeeze()
                    current_sil =  current_sil[0:1024, 0:1280]
                    current_GT_sil = (silhouette[i]/255).type(torch.FloatTensor).to(device)

                    if(traintype == 'render'):

                        if useofFK: #use of the mlp
                            print('render with FK for Rt')
                            if (i == 0):
                                loss = nn.BCELoss()(current_sil, current_GT_sil).to(device)
                            else:
                                loss += nn.BCELoss()(current_sil, current_GT_sil).to(device)

                        else: #use sigmoid curve
                            if (i == 0):
                                print('render with sigmoid for Rt')
                                loss = (nn.BCELoss()(current_sil, current_GT_sil).to(device))*(1 - alpha) + (nn.MSELoss()(params[i], parameter[i]).to(device))*(alpha)
                            else:
                                loss += (nn.BCELoss()(current_sil, current_GT_sil).to(device)) * (1 - alpha) + (nn.MSELoss()(params[i], parameter[i]).to(device)) *(alpha)


                    if(traintype == 'regression'):
                        print('regression for Rt')
                        if (i == 0):
                            loss =nn.MSELoss()(params[i], parameter[i]).to(device)
                        else:
                            loss = loss + nn.MSELoss()(params[i], parameter[i]).to(device)
                        print(loss)

                if ResnetOutput == 't':
                    print('Resnet output 3 values')
                    model.t = t_params[i]
                    print(model.t)
                    R = parameter[i, 0:3] #give the ground truth parameter for the rotation values
                    model.R = R2Rmat(R)
                    current_sil = model.renderer(model.vertices, model.faces, R=model.R, t=model.t,
                                                 mode='silhouettes').squeeze()
                    current_sil = current_sil[0:1024, 0:1280]
                    current_GT_sil = (silhouette[i] / 255).type(torch.FloatTensor).to(device)

                    if (traintype == 'render'):
                        print('t_param is {}'.format(t_params[i]))
                        print('Gt_param is {}'.format(parameter[i, 3:6]))

                        if useofFK:  # use of the mlp
                            print('render with FK for t')
                            if (i == 0):
                                loss = nn.BCELoss()(current_sil, current_GT_sil).to(device)
                            else:
                                loss += nn.BCELoss()(current_sil, current_GT_sil).to(device)

                        else:  # use sigmoid curve
                            print('render with sigmoid for t')
                            test = (nn.BCELoss()(current_GT_sil, current_GT_sil).to(device))
                            print('bce of 2 same picture is {}'.format(test))
                            if (i == 0):
                                loss = (nn.BCELoss()(current_sil, current_GT_sil).to(device)) * (1 - alpha) + (nn.MSELoss()(t_params[i], parameter[i, 3:6]).to(device)) * (alpha)
                            else:
                                loss += (nn.BCELoss()(current_sil, current_GT_sil).to(device)) * (1 - alpha) + (nn.MSELoss()(t_params[i], parameter[i, 3:6]).to(device)) * (alpha)

                    if (traintype == 'regression'):
                        print('regression for t')
                        if (i == 0):
                            print('t_param is {}'.format(t_params[i]))
                            print('Gt_param is {}'.format(parameter[i, 3:6]))
                            loss = nn.MSELoss()(t_params[i], parameter[i, 3:6]).to(device)
                        else:
                            loss = loss + nn.MSELoss()(t_params[i], parameter[i, 3:6]).to(device)




            loss = loss / numbOfImage
            print('number of image is {}'.format(numbOfImage))
            print('step {} loss is {}'.format(count,loss))
            loss.backward()
            optimizer.step()

            current_step_Train_loss .append(loss.detach().cpu().numpy())  # contain only this epoch loss, will be reset after each epoch
            count = count + 1

        epochTrainloss = np.mean(current_step_Train_loss)
        epochsTrainLoss.write('step: {}/{} current step loss: {:.4f}\r\n'.format(epoch, n_epochs, epochTrainloss))
        print('loss of epoch {} is {}'.format(epoch, epochTrainloss))
        current_step_Train_loss  = [] #reset value
        Epoch_Train_losses.append(epochTrainloss)  # most significant value to store

        ## test phase --------------------------------------------------------------------------------------------------------
        count = 0
        testcount = 0

        model.eval()

        current_step_Test_loss = []

        steps_losses = []  # reset the list after each epoch
        steps_alpha_loss = []
        steps_beta_loss = []
        steps_gamma_loss = []
        steps_x_loss = []
        steps_y_loss = []
        steps_z_loss = []

        for i in range(number_test_im):
            output_test_image_dir = '{}/images/testIm{}'.format(output_result_dir, i)
            mkdir_p(output_test_image_dir)

        t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))
        for image, silhouette, parameter in t:

            Test_Step_loss = []
            numbOfImage = image.size()[0]



            image = image.to(device)



            parameter = parameter.to(device)

            if ResnetOutput == 'Rt':
                params = model(image)
            if ResnetOutput == 't':
                t_params = model(image)  # should be
                # print(t_params.size())

            # params = model(image, 0, False)  # should be size [batchsize, 6]
            # print(np.shape(params))


            for i in range(0, numbOfImage):

                if ResnetOutput == 'Rt':
                    print('test for Rt')

                    print('image tested: {}'.format(testcount))
                    print('estimated {}'.format(params[i]))
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

                    model.t = params[i, 3:6]
                    R = params[i, 0:3]
                    model.R = R2Rmat(R)  # angle from resnet are in radian

                if ResnetOutput == 't':
                    print('test for t')
                    print('image tested: {}'.format(testcount))
                    print('estimated {}'.format(t_params[i]))
                    print('Ground Truth {}'.format(parameter[i, 3:6]))
                    if (i == 0):
                        loss = nn.MSELoss()(t_params[i], parameter[i, 3:6]).to(device)
                    else:
                        loss = loss + nn.MSELoss()(t_params[i], parameter[i, 3:6]).to(device)

                    # one value each for the step, compute mse loss for all parameters separately
                    alpha_loss = 0
                    beta_loss = 0
                    gamma_loss = 0
                    x_loss = nn.MSELoss()(t_params[:, 0], parameter[:, 3]).detach().cpu().numpy()
                    y_loss = nn.MSELoss()(t_params[:, 1], parameter[:, 4]).detach().cpu().numpy()
                    z_loss = nn.MSELoss()(t_params[:, 2], parameter[:, 5]).detach().cpu().numpy()

                    steps_losses.append(loss.item())  # only one loss value is add each step
                    steps_alpha_loss.append(0)
                    steps_beta_loss.append(0)
                    steps_gamma_loss.append(0)
                    steps_x_loss.append(x_loss.item())
                    steps_y_loss.append(y_loss.item())
                    steps_z_loss.append(z_loss.item())

                    model.t = t_params[i]
                    R = parameter[i, 0:3] #give the ground truth parameter for the rotation values
                    model.R = R2Rmat(R)
            print('Rt test GTparameter are {}'.format(parameter))
            print('Rt test parameter are {}{}'.format(R, model.t))
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
            plt.savefig('{}/images/testIm{}/im{}epoch{}.png'.format(output_result_dir, testcount, testcount, epoch), bbox_inches='tight',
                        pad_inches=0.05)
            plt.show()
            plt.close()

            #MSE loss
            current_step_Test_loss.append(loss.detach().cpu().numpy())
            testcount = testcount + 1
     #loop here  for each epoch

        epoch_test_loss.append(np.mean(steps_losses))
        epoch_test_alpha_loss.append(np.mean(steps_alpha_loss))
        epoch_test_beta_loss.append(np.mean(steps_beta_loss))
        epoch_test_gamma_loss.append(np.mean(steps_gamma_loss))
        epoch_test_x_loss.append(np.mean(steps_x_loss))
        epoch_test_y_loss.append(np.mean(steps_y_loss))
        epoch_test_z_loss.append(np.mean(steps_z_loss))

        TestParamLoss.write(
            'test epoch: {} current step loss: {:.7f}, angle loss: {:.4f} {:.4f} {:.4f} translation loss: {:.7f} {:.7f} {:.7f}\r\n'
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
    ax1.semilogy(Epoch_Train_losses)
    ax1.set_ylabel('train {} loss'.format(traintype))
    ax1.set_xlim([0, n_epochs])
    # ax1.set_ylim([0, 4])
    # ax1.set_yscale('log')


    ax2.semilogy(Epoch_Test_losses)
    ax2.set_ylabel('test {} loss'.format(traintype))
    ax2.set_xlabel('epoch')
    ax2.set_xlim([0, n_epochs])
    # ax2.set_ylim([0, 0.1])
    # ax2.set_yscale('log')


    plt.savefig('{}/training_epochs_{}_results_{}.png'.format(output_result_dir,traintype,fileExtension), bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()

    fig, (a,b,g) = plt.subplots(3, 1)
    a.semilogy(epoch_test_alpha_loss)
    a.set_xlim([0, n_epochs])
    a.set_ylabel('a')

    b.semilogy(epoch_test_beta_loss)
    b.set_xlim([0, n_epochs])
    b.set_ylabel('b')

    g.semilogy(epoch_test_gamma_loss)
    g.set_xlim([0, n_epochs])
    g.set_ylabel('g')
    g.set_xlabel('MSE loss evolution through epochs')

    plt.savefig('{}/test_epochs_{}_R_Loss_{}.png'.format(output_result_dir,traintype,fileExtension), bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()

    fig, (x,y,z) = plt.subplots(3, 1)

    x.semilogy(epoch_test_x_loss)
    x.set_xlim([0, n_epochs])
    x.set_ylabel('x')

    y.semilogy(epoch_test_y_loss)
    y.set_xlim([0, n_epochs])
    y.set_ylabel('y')

    z.semilogy(epoch_test_z_loss)
    z.set_xlim([0, n_epochs])
    z.set_ylabel('z')
    z.set_xlabel('MSE loss evolution through epochs')


    plt.savefig('{}/test_epochs_{}_t_Loss_{}.png'.format(output_result_dir,traintype,fileExtension), bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()


    # save the model
    output_model_dir = '{}/model'.format(output_result_dir)
    mkdir_p(output_model_dir)

    torch.save(model.state_dict(),'{}/FinalModel_train_{}.pth'.format(output_model_dir,fileExtension))
    print('parameters saved')

    epochsTrainLoss.close()
    TestParamLoss .close()


    # if validation:
    #     # validation --------------------------------------------------------------------------------------------------------
    #
    #     print('validation phase')
    #     Valcount = 0
    #     processcount = 0
    #     current_step_Val_loss = []
    #     Epoch_Val_losses = []
    #     from PIL import ImageTk, Image, ImageDraw
    #     epochsValLoss = open("./results/valNoParamShift.txt", "w+")
    #
    #     t = tqdm(iter(val_dataloader), leave=True, total=len(val_dataloader))
    #     for image, silhouette, parameter in t:
    #         Test_Step_loss = []
    #         numbOfImage = image.size()[0]
    #         # image1 = torch.flip(image,[0, 3]) #flip vertical
    #         # image = torch.roll(image, 100, 3) #shift down from 100 px
    #         # image1 = shiftPixel(image, 100, 'y')
    #         image1 = image
    #         image1 = image1.to(device)  # torch.Size([1, 3, 1024, 1280])
    #         parameter = parameter.to(device)
    #         params = model(image1)  # should be size [batchsize, 6]
    #         # print(np.shape(params))
    #         i = 0
    #
    #         # print('image estimated: {}'.format(testcount))
    #         # print('estimated parameter {}'.format(params[i]))
    #         # print('Ground Truth {}'.format(parameter[i]))
    #
    #         epochsValLoss.write('step:{} params:{} \r\n'.format(processcount, params.detach().cpu().numpy()))
    #         loss = nn.MSELoss()(params[i], parameter[i]).to(device)
    #
    #         model.t = params[i, 3:6]
    #         R = params[i, 0:3]
    #         model.R = R2Rmat(R)  # angle from resnet are in radian
    #
    #         current_sil = model.renderer(model.vertices, model.faces, R=model.R, t=model.t,
    #                                      mode='silhouettes').squeeze()
    #         current_sil = current_sil[0:1024, 0:1280]
    #
    #         sil2plot = np.squeeze((current_sil.detach().cpu().numpy() * 255)).astype(np.uint8)
    #
    #         image2show = np.squeeze((image1[i].detach().cpu().numpy()))
    #         image2show = (image2show * 0.5 + 0.5).transpose(1, 2, 0)
    #         # image2show = np.flip(image2show,1)
    #
    #         # fig = plt.figure()
    #         # fig.add_subplot(2, 1, 1)
    #         # plt.imshow(sil2plot, cmap='gray')
    #         #
    #         # fig.add_subplot(2, 1, 2)
    #         # plt.imshow(image2show)
    #         # plt.show()
    #
    #         sil3d = sil2plot[:, :, np.newaxis]
    #         renderim = np.concatenate((sil3d, sil3d, sil3d), axis=2)
    #         toolIm = Image.fromarray(np.uint8(renderim))
    #
    #         # print(type(image2show))
    #         backgroundIm = Image.fromarray(np.uint8(image2show * 255))
    #
    #         # backgroundIm.show()
    #         #
    #
    #         alpha = 0.2
    #         out = Image.blend(backgroundIm, toolIm, alpha)
    #         # #make gif
    #         imsave('/tmp/_tmp_%04d.png' % processcount, np.array(out))
    #         processcount = processcount + 1
    #         # print(processcount)
    #
    #     print('making the gif')
    #     make_gif(args.filename_output)
    #     current_step_Val_loss.append(loss.detach().cpu().numpy())
    #     epochsValLoss.close()