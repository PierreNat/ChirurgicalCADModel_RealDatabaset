import numpy as np
import math as m
import torch


def BuildTransformationMatrix(tx=0, ty=0, tz=0, alpha=0, beta=0, gamma=0):

    # alpha = alpha - pi/2
    alpha = alpha
    beta = beta
    gamma = gamma


    Rx = np.array([[1, 0, 0],
                   [0, m.cos(alpha), -m.sin(alpha)],
                   [0, m.sin(alpha), m.cos(alpha)]])

    Ry = np.array([[m.cos(beta), 0, m.sin(beta)],
                   [0, 1, 0],
                   [-m.sin(beta), 0, m.cos(beta)]])

    Rz = np.array([[m.cos(gamma), -m.sin(gamma), 0],
                   [m.sin(gamma), m.cos(gamma), 0],
                   [0, 0, 1]])


# create the rotation object matrix

    Rzy = np.matmul(Rz, Ry)
    Rzyx = np.matmul(Rzy, Rx)

    # R = np.matmul(Rx, Ry)
    # R = np.matmul(R, Rz)

    t = torch.from_numpy(np.array([tx, ty, tz]).astype(np.float32))


    return t, Rzyx


class camera_setttings():

    # define extrinsinc and instrinsic camera parameter
    def __init__(self, R= np.array([np.radians(0), np.radians(0), np.radians(0)]), t=np.array([0, 0, 0]), PnPtm= 0, PnPtmFlag = False,
                 vert=1, resolutionx=0, resolutiony=0, cx=0, cy=0, fx=0, fy=0): #R 1x3 array, t 1x2 array, number of vertices

        self.R = R
        self.t = t
        self.PnPtm = PnPtm
        self.alpha = R[0]
        self.beta= R[1]
        self.gamma = R[2]
        self.tx = t[0]
        self.ty = t[1]
        self.tz = t[2]

        self.resolutionX = resolutionx  # in pixel
        self.resolutionY = resolutiony



        # K from file
        self.c_x = cx # 1080
        self.c_y = cy # 1016
        self.f_x = fx
        self.f_y = fy

        self.K = np.array([[self.f_x, 0, self.c_x],
                      [0, self.f_y, self.c_y],
                      [0, 0, 1]])  # shape of [nb_vertice, 3, 3]

        # angle in radian
        if PnPtmFlag:
            # self.t_mat = torch.from_numpy(np.array([PnPtm[0,3], PnPtm[1,3], PnPtm[2,3]]).astype(np.float32))
            # self.R_mat = PnPtm[0:3,0:3]

            self.t_mat2, self.R_mat2 = BuildTransformationMatrix(self.tx, self.ty, self.tz, self.alpha, self.beta,
                                                               self.gamma)

        else:

            self.t_mat, self.R_mat = BuildTransformationMatrix(self.tx, self.ty, self.tz, self.alpha, self.beta,
                                                               self.gamma)

        self.K_vertices = np.repeat(self.K[np.newaxis, :, :], vert, axis=0)
        self.R_vertices = np.repeat(self.R_mat[np.newaxis, :, :], vert, axis=0)
        self.t_vertices = np.repeat(self.t_mat[np.newaxis, :], 1, axis=0).cuda()
        self.t_vertices.requires_grad = True