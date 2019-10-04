
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

from scipy import linalg
# https://www.oreilly.com/library/view/programming-computer-vision/9781449341916/ch04.html
class Camera(object):
  """ Class for representing pin-hole cameras. """

  def __init__(self,P):
    """ Initialize P = K[R|t] camera model. """
    self.P = P
    self.K = None # calibration matrix
    self.R = None # rotation camera
    self.t = None # translation camera
    self.c = None # camera center

    c_x = 590.04 #1080
    c_y = 508.74 #1016
    f_x = 1067.70
    f_y = 1067.52


    self.K  = np.array([[f_x,0,c_x],
                  [0,f_y,c_y],
                  [0,0,1]])  # shape of [nb_vertice, 3, 3]


  def project(self,X):
    """  Project points in X (4*n array) and normalize coordinates. """

    x = np.dot(self.P,X)
    for i in range(3):
      x[i] /= x[2]
    return x


def main():


    nb_im = 1

    ## -------------------------read json file -------------------------------------------

    with open('data/data.json') as json_file:
        data = json.load(json_file)
        data_len = len(data)
        # usm_camera = data[0:data_len]['usm-1']

    ## --------------------------------------------------------------------------------

    loop = tqdm.tqdm(range(0, nb_im))
    for i in loop:

        ## -------------------------extract json frame matrix -------------------------------------------
        usm_camera = data[i]['usm-1']
        usm_inst = data[i]['usm-2']


        instrument_to_camera_transform = np.asarray([list(map(float, usm_inst['pose'][0])),
                                                     list(map(float, usm_inst['pose'][1])),
                                                     list(map(float, usm_inst['pose'][2])),
                                                     list(map(float, usm_inst['pose'][3]))],
                                                    dtype=np.float64)

        #to test the conversion degree to radian to transformation matrix and then back to euler angle in radian
        R_test = np.array([np.radians(0),np.radians(0),np.radians(0)]) #test value alpha beta gamma
        T_test_vector, R_test_matrix =  BuildTransformationMatrix(tx=0, ty=0, tz=0, alpha=R_test[0], beta=R_test[1], gamma=R_test[2])
        instrument_to_camera_transform[0,0:3] = R_test_matrix[0,:]
        instrument_to_camera_transform[1,0:3] = R_test_matrix[1,:]
        instrument_to_camera_transform[2,0:3] = R_test_matrix[2,:]


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

        # create a point
        point = [0,0,0,0]

        # setup camera
        P = hstack((eye(3),array([[0],[0],[-10]])))
        cam = camera.Camera(P)
        x = cam.project(points)

        # plot projection
        figure()
        plot(x[0],x[1],'k.')
        show()

if __name__ == '__main__':
    main()
