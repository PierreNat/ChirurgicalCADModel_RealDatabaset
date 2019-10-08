
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
import math as m
import argparse
import imageio
import json
import glob
from skimage.io import imread, imsave
import matplotlib.image as mpimg
import random
from scipy.misc import imsave
from utils_functions.R2Rmat import R2Rmat
from utils_functions.camera_settings import camera_setttings
from utils_functions.camera_settings import BuildTransformationMatrix
import torch
from numpy.random import uniform
import neural_renderer as nr

def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()

def main():


    nb_im = 100
    space = 1

    ## -------------------------read json file -------------------------------------------

    with open('data/data.json') as json_file:
        data = json.load(json_file)
        data_len = len(data)
        # usm_camera = data[0:data_len]['usm-1']

    ## --------------------------------------------------------------------------------

    loop = tqdm.tqdm(range(0, nb_im))
    All2D_point = []
    AllOrigin2D_point = []
    All_X_2D_point = []
    All_Y_2D_point = []
    All_Z_2D_point = []
    All2D_point2 = []
    FrameNumb = []
    for i in loop:
        frame= i*space
        print(frame)
        FrameNumb.append(frame)
        ## -------------------------extract json frame matrix -------------------------------------------
        usm_camera = data[frame]['usm-1']
        usm_inst = data[frame]['usm-2']


        instrument_to_camera_transform = np.asarray([list(map(float, usm_inst['pose'][0])),
                                                     list(map(float, usm_inst['pose'][1])),
                                                     list(map(float, usm_inst['pose'][2])),
                                                     list(map(float, usm_inst['pose'][3]))],
                                                    dtype=np.float64)
        # print(instrument_to_camera_transform) #[4x4]
        # instrument_to_camera_transform = instrument_to_camera_transform[0:3,0:4] #3x4matrix
        # create a point
        length = 0.3
        Origin_point_3D = np.array((0,0,0,1))
        # point2_3D = np.array((0,0,0.3,1))
        Xaxis_point_3D = np.array((length,0,0,1))
        Yaxis_point_3D = np.array((0,length,0,1))
        Zaxis_point_3D = np.array((0,0,length,1))

        # setup camera
        c_x = 590.04  # 1080
        c_y = 508.74  # 1016
        f_x = 1067.70
        f_y = 1067.52

        K = np.array([[f_x, 0, c_x,0],
                       [0, f_y, c_y,0],
                       [0, 0, 1,0]])  # [3x4]



        point_in_camera = np.matmul(instrument_to_camera_transform, Origin_point_3D)
        point_2d = np.matmul(K,point_in_camera)
        point_2d = point_2d/point_2d[2]
        AllOrigin2D_point.append((point_2d))

        point_in_camera = np.matmul(instrument_to_camera_transform, Xaxis_point_3D)
        point_2d = np.matmul(K,point_in_camera)
        point_2d = point_2d/point_2d[2]
        All_X_2D_point.append((point_2d))

        point_in_camera = np.matmul(instrument_to_camera_transform, Yaxis_point_3D)
        point_2d = np.matmul(K,point_in_camera)
        point_2d = point_2d/point_2d[2]
        All_Y_2D_point.append((point_2d))

        point_in_camera = np.matmul(instrument_to_camera_transform, Zaxis_point_3D)
        point_2d = np.matmul(K,point_in_camera)
        point_2d = point_2d/point_2d[2]
        All_Z_2D_point.append((point_2d))


        # #to test the conversion degree to radian to transformation matrix and then back to euler angle in radian
        # R_test = np.array([np.radians(0),np.radians(0),np.radians(0)]) #test value alpha beta gamma
        # T_test_vector, R_test_matrix =  BuildTransformationMatrix(tx=0, ty=0, tz=0, alpha=R_test[0], beta=R_test[1], gamma=R_test[2])
        # instrument_to_camera_transform[0,0:3] = R_test_matrix[0,:]
        # instrument_to_camera_transform[1,0:3] = R_test_matrix[1,:]
        # instrument_to_camera_transform[2,0:3] = R_test_matrix[2,:]


        joint_values = np.asarray([list(map(float, usm_inst['articulation'][0])),
                                   list(map(float, usm_inst['articulation'][1])),
                                   list(map(float, usm_inst['articulation'][2]))],
                                  dtype=np.float64)

        joint_values[-1] = 2 * joint_values[-1]
        # print(instrument_to_camera_transform[1,2]) # [row column]



        #formula from http://planning.cs.uiuc.edu/node103.html
        # alpha and gamma were swapped, don-t know where the problem is but value are correct
        Extracted_theta3_rad = m.atan2(instrument_to_camera_transform[1,0],instrument_to_camera_transform[0,0])
        C_2 = m.sqrt(instrument_to_camera_transform[2,1]*instrument_to_camera_transform[2,1] + instrument_to_camera_transform[2,2]*instrument_to_camera_transform[2,2])
        Extracted_theta2_rad = m.atan2(-instrument_to_camera_transform[2,0],  C_2 )
        Extracted_theta1_rad = m.atan2(instrument_to_camera_transform[2,1],instrument_to_camera_transform[2,2])

        #formula from https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/07/euler-angles1.pdf
        # Extracted_theta1_rad = m.atan2(-instrument_to_camera_transform[1,2],instrument_to_camera_transform[2,2])
        # C_2 = m.sqrt(instrument_to_camera_transform[0,0]*instrument_to_camera_transform[0,0] + instrument_to_camera_transform[0,1]*instrument_to_camera_transform[0,1])
        # Extracted_theta2_rad = m.atan2(instrument_to_camera_transform[0,1],  C_2 )
        # s_1 = m.sin(Extracted_theta1_rad)
        # c_1 = m.cos(Extracted_theta1_rad)
        # Extracted_theta3_rad = m.atan2(s_1*instrument_to_camera_transform[2,0]-c_1*instrument_to_camera_transform[1,0],
        #                                c_1*instrument_to_camera_transform[1,1]-s_1*instrument_to_camera_transform[2,1])


        Extracted_X =  instrument_to_camera_transform[0,3]
        Extracted_Y =  instrument_to_camera_transform[1,3]
        Extracted_Z =  instrument_to_camera_transform[2,3]

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

    print(len(FrameNumb))

    # #plot scatter point with line
    # for i in range(0,len(FrameNumb)):
    #     motion = plt.scatter(x=[AllOrigin2D_point[i][0]], y=[AllOrigin2D_point[i][1]-1024], c='k', s=10)
    #     motion = plt.scatter(x=[All_X_2D_point[i][0]], y=[All_X_2D_point[i][1]-1024], c='r', s=10)
    #     motion = plt.scatter(x=[All_Y_2D_point[i][0]], y=[All_Y_2D_point[i][1]-1024], c='g', s=10)
    #     motion = plt.scatter(x=[All_Z_2D_point[i][0]], y=[All_Z_2D_point[i][1]-1024], c='b', s=10)
    #     plt.plot([AllOrigin2D_point[i][0], All_X_2D_point[i][0]], [AllOrigin2D_point[i][1]-1024, All_X_2D_point[i][1]-1024], 'r-')
    #     plt.plot([AllOrigin2D_point[i][0], All_Y_2D_point[i][0]], [AllOrigin2D_point[i][1]-1024, All_Y_2D_point[i][1]-1024], 'g-')
    #     plt.plot([AllOrigin2D_point[i][0], All_Z_2D_point[i][0]], [AllOrigin2D_point[i][1]-1024, All_Z_2D_point[i][1]-1024], 'b-')
    #     plt.xlabel("X")
    #     plt.ylabel("Y")
    #     plt.axis([0, 1280, 0, -1024])
    # plt.show()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_name_extension = 'test_{}images'.format(nb_im)
    result_dir = os.path.join(current_dir, 'framesgif')
    parser = argparse.ArgumentParser()
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(result_dir, 'GIF_{}.gif'.format(file_name_extension)))
    args = parser.parse_args()


    loop = tqdm.tqdm(range(0,len(FrameNumb)))
    for i in loop:
        # plt.subplot(1,len(FrameNumb),i+1)
        im = plt.imread('framesLeft/frameL{}.jpg'.format(FrameNumb[i]))
        plt.imshow(im)
        plt.scatter(x=[AllOrigin2D_point[i][0]], y=[AllOrigin2D_point[i][1]], c='k', s=10)
        plt.scatter(x=[All_X_2D_point[i][0]], y=[All_X_2D_point[i][1]], c='r', s=10)
        plt.scatter(x=[All_Y_2D_point[i][0]], y=[All_Y_2D_point[i][1]], c='g', s=10)
        plt.scatter(x=[All_Z_2D_point[i][0]], y=[All_Z_2D_point[i][1]], c='b', s=10)
        plt.plot([AllOrigin2D_point[i][0], All_X_2D_point[i][0]], [AllOrigin2D_point[i][1], All_X_2D_point[i][1]], 'r-')
        plt.plot([AllOrigin2D_point[i][0], All_Y_2D_point[i][0]], [AllOrigin2D_point[i][1], All_Y_2D_point[i][1]], 'g-')
        plt.plot([AllOrigin2D_point[i][0], All_Z_2D_point[i][0]], [AllOrigin2D_point[i][1], All_Z_2D_point[i][1]], 'b-')
        # plt.scatter(x=20, y=100, c='r', s=40)
        plt.savefig('framesgif/xyz{}.jpg'.format(i))

        plt.savefig('/tmp/_tmp_%04d.png' % i)
        plt.clf()

    # plt.show()
# save database
    print('making gif')
    make_gif(args.filename_output)


if __name__ == '__main__':
    main()
