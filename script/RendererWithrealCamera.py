
import numpy as np
import torch
from numpy.random import uniform
import neural_renderer as nr
import matplotlib.pyplot as plt
from utils_functions.camera_settings import BuildTransformationMatrix
import tqdm
import math as m
import matplotlib.image as mpimg
import random
from scipy.misc import imsave
from utils_functions.R2Rmat import R2Rmat
import imageio
import json
from utils_functions.camera_settings import camera_setttings

def main():

    cubes_database = []
    sils_database = []
    params_database = []
    im_nr = 1

    vertices_1, faces_1, textures_1 = nr.load_obj("3D_objects/rubik_color.obj", load_texture=True) #, texture_size=4)
    print(vertices_1.shape)
    print(faces_1.shape)
    vertices_1 = vertices_1[None, :, :]  # add dimension
    faces_1 = faces_1[None, :, :]  #add dimension
    textures_1 = textures_1[None, :, :]  #add dimension
    nb_vertices = vertices_1.shape[0]

    print(vertices_1.shape)
    print(faces_1.shape)

    file_name_extension = 'shaft_1im_180_M15_15_5_7'


    nb_im = 20
    useTransformMatrix = True
    instrument_to_camera_transform = np.array([(0, 0, 0, 0),
                                               (0, 0, 0, 0),
                                               (0, 0, 0, 0),
                                               (0, 0, 0, 1)])
    instrument_to_camera_transform = np.zeros((4,4))

    ## --------------------------------------------------------------------------------

    loop = tqdm.tqdm(range(0, nb_im))
    for i in loop:




        # # define transfomration parameter from angle and translation
        if useTransformMatrix :
            R_test = np.array([np.radians(i*15), np.radians(0), np.radians(0)])  # test value alpha beta gamma
            T_test = np.array([0, 0, 6])
            T_test_vector, R_test_matrix = BuildTransformationMatrix(tx=T_test[0], ty=T_test[1], tz=T_test[2], alpha=R_test[0], beta=R_test[1],
                                                                     gamma=R_test[2]) #return 1x3 vector and 3x3matrix
            instrument_to_camera_transform[0, 0:3] = R_test_matrix[0, :]
            instrument_to_camera_transform[1, 0:3] = R_test_matrix[1, :]
            instrument_to_camera_transform[2, 0:3] = R_test_matrix[2, :]
            instrument_to_camera_transform[0, 3] = T_test[0]
            instrument_to_camera_transform[1, 3] = T_test[1]
            instrument_to_camera_transform[2, 3] = T_test[2]
            instrument_to_camera_transform[3, 3] = 1
            # test[0:3,0:3] = R_test_matrix

            # formula from http://planning.cs.uiuc.edu/node103.html
            # alpha and gamma were swapped, don-t know where the problem is but value are correct
            Extracted_theta3_rad = m.atan2(instrument_to_camera_transform[1, 0], instrument_to_camera_transform[0, 0])
            C_2 = m.sqrt(instrument_to_camera_transform[2, 1] * instrument_to_camera_transform[2, 1] +
                         instrument_to_camera_transform[2, 2] * instrument_to_camera_transform[2, 2])
            Extracted_theta2_rad = m.atan2(-instrument_to_camera_transform[2, 0], C_2)
            Extracted_theta1_rad = m.atan2(instrument_to_camera_transform[2, 1], instrument_to_camera_transform[2, 2])


            Extracted_X = instrument_to_camera_transform[0, 3]
            Extracted_Y = instrument_to_camera_transform[1, 3]
            Extracted_Z = instrument_to_camera_transform[2, 3]

            Extracted_theta1_deg = np.degrees(Extracted_theta1_rad)
            Extracted_theta2_deg = np.degrees(Extracted_theta2_rad)
            Extracted_theta3_deg = np.degrees(Extracted_theta3_rad)

            # define transfomration parameter from json file
            alpha =Extracted_theta1_deg
            beta = Extracted_theta2_deg
            gamma =  Extracted_theta3_deg
            x = Extracted_X
            y = Extracted_Y
            z = Extracted_Z
            print(x,y,z, alpha,beta,gamma)

        else:
            alpha =145#uniform(0, 180)
            beta = 5.2#uniform(0, 180)
            gamma =  -89 #uniform(0, 180)
            x = 0#uniform(-1.5, 1.5)
            y =0 #uniform(-1.5, 1.5)
            z = 1#uniform(5, 7) #1000t was done with value between 7 and 10, Rot and trans between 5 10



        R = np.array([np.radians(alpha), np.radians(beta), np.radians(gamma)])  # angle in degree
        t = np.array([x, y, z])  # translation in meter

        # define transformation by transformation matrix

        Rt = np.concatenate((R, t), axis=None).astype(np.float16)  # create one array of parameter in radian, this arraz will be saved in .npy file

        cam = camera_setttings(R=R, t=t, vert=nb_vertices, resolutionx=1280, resolutiony=1024,cx=590, cy=508, fx=1067, fy=1067) # degree angle will be converted  and stored in radian

        renderer = nr.Renderer(image_size=1280, camera_mode='projection', dist_coeffs=None,anti_aliasing=False,
                               K=cam.K_vertices, R=cam.R_vertices, t=cam.t_vertices, near=0.01, background_color=[1, 1, 1],
                               # background is filled now with  value 0-1 instead of 0-255
                               # changed from 0-255 to 0-1
                               far=10, orig_size=500,
                               light_intensity_ambient=1, light_intensity_directional=1, light_direction=[0, 1, 0],
                               light_color_ambient=[1, 1, 1], light_color_directional=[1, 1, 1])

        images_1 = renderer(vertices_1, faces_1, textures_1,
                            K=torch.cuda.FloatTensor(cam.K_vertices),
                            R=torch.cuda.FloatTensor(cam.R_vertices),
                            t=torch.cuda.FloatTensor(cam.t_vertices))  # [batch_size, RGB, image_size, image_size]

        image = images_1[0].detach().cpu().numpy()[0].transpose((1, 2, 0)) #float32 from 0-1
        image = (image*255).astype(np.uint8) #cast from float32 255.0 to 255 uint8

        sils_1 = renderer(vertices_1, faces_1, textures_1,
                          mode='silhouettes',
                          K=torch.cuda.FloatTensor(cam.K_vertices),
                          R=torch.cuda.FloatTensor(cam.R_vertices),
                             t=torch.cuda.FloatTensor(cam.t_vertices))  # [batch_size, RGB, image_size, image_size]

        sil = sils_1.detach().cpu().numpy().transpose((1, 2, 0))
        sil = np.squeeze((sil * 255)).astype(np.uint8) # change from float 0-1 [512,512,1] to uint8 0-255 [512,512]

        #grow the list of cube, silhouette and parameters
        cubes_database.extend(image)
        sils_database.extend(sil)
        params_database.extend(Rt)

        im_nr = im_nr+1


        if(im_nr%1 == 0):
            fig = plt.figure()
            fig.add_subplot(2, 1, 1)
            plt.imshow(image)
            imageio.imwrite("3D_objects/{}_ref.png".format(file_name_extension), image)

            fig.add_subplot(2, 1, 2)
            plt.imshow(sil, cmap='gray')
            plt.show()
            plt.close(fig)

# save database
# # reshape in the form (nbr of image, x dim, y dim, layers)
#     cubes_database = np.reshape(cubes_database, (im_nr-1, 512, 512, 3)) # 3 channel rgb
#     sils_database = np.reshape(sils_database, (im_nr-1, 512, 512)) #binary mask monochannel
#     params_database = np.reshape(params_database,(im_nr-1, 6)) #array of 6 params
#     np.save('Npydatabase/cubes_{}.npy'.format(file_name_extension), cubes_database)
#     np.save('Npydatabase/sils_{}.npy'.format(file_name_extension), sils_database)
#     np.save('Npydatabase/params_{}.npy'.format(file_name_extension), params_database)
#     print('images saved')


if __name__ == '__main__':
    main()
