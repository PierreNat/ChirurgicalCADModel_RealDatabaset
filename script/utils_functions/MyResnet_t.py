import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from skimage.io import imread, imsave
import torch
from torchvision.models.resnet import ResNet, Bottleneck
from utils_functions.camera_settings import camera_setttings
import math as m
import torch.utils.model_zoo as model_zoo
import neural_renderer as nr
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def Myresnet50_t(filename_obj=None, pretrained=True, cifar = True, modelName='None', **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ModelResNet50( filename_obj=filename_obj)
    if pretrained:
        print('using own pre-trained model')

        if cifar == True:
            pretrained_state = model_zoo.load_url(model_urls['resnet50'])
            model_state = model.state_dict()
            pretrained_state = {k: v for k, v in pretrained_state.items() if
                                k in model_state and v.size() == model_state[k].size()}
            model_state.update(pretrained_state)
            model.load_state_dict(model_state)
            model.eval()

        else:
            model.load_state_dict(torch.load('models/{}.pth'.format(modelName)))
            model.eval()
        print('download finished')
    return model




class ModelResNet50(ResNet):
    def __init__(self, filename_obj=None, *args, **kwargs):
        super(ModelResNet50, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=3, **kwargs)

# resnet part
        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        )

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        )

        self.fc

        # self.fc1 = nn.Linear(2*6, 6)
        # self.fc2 = nn.Linear(8, 6)


# render part

        vertices, faces, textures = nr.load_obj(filename_obj, load_texture=True, normalization=False)
        vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
        faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3
        textures = textures[None, :, :]
        nb_vertices = vertices.shape[0]

        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)
        self.register_buffer('textures', textures)


        # ---------------------------------------------------------------------------------
        # extrinsic parameter, link world/object coordinate to camera coordinate
        # ---------------------------------------------------------------------------------

        alpha = np.radians(0)
        beta = np.radians(0)
        gamma = np.radians(0)

        x = 0  # uniform(-2, 2)
        y = 0  # uniform(-2, 2)
        z = 0.08  # uniform(5, 10) #1000t was done with value between 7 and 10, Rot and trans between 5 10

        R = np.array([np.radians(alpha), np.radians(beta), np.radians(gamma)])  # angle in degree
        t = np.array([x, y, z])  # translation in meter

        batch = vertices.shape[0]

        # ---------------------------------------------------------------------------------
        # intrinsic parameter
        # ---------------------------------------------------------------------------------
        c_x = 590
        c_y = 508

        f_x = 1067
        f_y = 1067

        # camera_calibration = np.zeros((4, 4))
        # camera_calibration[0, 0] = f_x
        # camera_calibration[1, 1] = f_y
        # camera_calibration[0, 2] = c_x
        # camera_calibration[1, 2] = c_y
        # camera_calibration[2, 2] = 1

        cam = camera_setttings(R=R, t=t, PnPtm = 0, PnPtmFlag = False, vert=nb_vertices, resolutionx=1280, resolutiony=1024,cx=c_x, cy=c_y, fx=f_x, fy=f_y) # degree angle will be converted  and stored in radian


        # K = np.array([[f / pix_sizeX, 0, Cam_centerX],
        #               [0, f / pix_sizeY, Cam_centerY],
        #               [0, 0, 1]])  # shape of [nb_vertice, 3, 3]
        #
        # K = np.repeat(K[np.newaxis, :, :], batch, axis=0)  # shape of [batch=1, 3, 3]
        # R = np.repeat(R[np.newaxis, :, :], batch, axis=0)  # shape of [batch=1, 3, 3]
        # t = np.repeat(t[np.newaxis, :], 1, axis=0)  # shape of [1, 3]
        #
        # self.K = K
        # self.R = R
        # # -------------------------- working block translation
        # self.tx = torch.from_numpy(np.array(x, dtype=np.float32)).cuda()
        # self.ty = torch.from_numpy(np.array(y, dtype=np.float32)).cuda()
        # self.tz = torch.from_numpy(np.array(z, dtype=np.float32)).cuda()
        # self.t =torch.from_numpy(np.array([self.tx, self.ty, self.tz], dtype=np.float32)).unsqueeze(0)

        # --------------------------

        # setup renderer
        renderer = nr.Renderer(image_size=1280, camera_mode='projection', dist_coeffs=None,anti_aliasing=True, fill_back=True, perspective=False,
                               K=cam.K_vertices, R=cam.R_vertices, t=cam.t_vertices, near=0, background_color=[1, 1, 1],
                               # background is filled now with  value 0-1 instead of 0-255
                               # changed from 0-255 to 0-1
                               far=1, orig_size=1280,
                               light_intensity_ambient=1, light_intensity_directional=0.5, light_direction=[0, 1, 0],
                               light_color_ambient=[1, 1, 1], light_color_directional=[1, 1, 1])

        self.renderer = renderer

    def forward(self, x, fkParam = 0, fk=False):
        # print('x has size {}'.format(x.size())) #x has size torch.Size([2, 3, 1024, 1280])
        # print('fk params are {}'.format(fkParam.size())) #x has size torch.Size([2, 6])
        x = self.seq1(x)
        x = self.seq2(x)
        # print(x.size()) # size is torch.Size([2, 2048, 1, 1])
        x = x.view(x.size(0), -1) #torch.Size([2, 2048])
        #  print(x.size())
        params = self.fc(x)
        # print('params {}'.format(params))

        if fk:
            third_tensor = torch.cat((params, fkParam), 1) #torch.Size([2, 12])
            # print('concat has size {}'.format(third_tensor.size()))
            third_tensor  = self.fc1(third_tensor)
            NewEstimate = third_tensor
            # print('NewEst has size {}'.format(NewEstimate.size()))
            # print('computed parameters are {}'.format(params))
            # return params
            # print('new estimage {}'.format(NewEstimate))
            return NewEstimate

        # print(params.size())
        else:
            # print('no fk')
            return params
