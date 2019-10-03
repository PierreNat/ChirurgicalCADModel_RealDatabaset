import json
import torch
import argparse
import os
import numpy as np
# import pysilico
import json
import numpy as np
import imageio

with open('data/data.json') as json_file:
    data = json.load(json_file)
    data_len = len(data)
    # usm_camera = data[0:data_len]['usm-1']
    usm_camera = data[0]['usm-1']
    usm_inst = data[0]['usm-2']

    instrument_to_camera_transform = np.asarray([list(map(float, usm_inst['pose'][0])),
                                                 list(map(float, usm_inst['pose'][1])),
                                                 list(map(float, usm_inst['pose'][2])),
                                                 list(map(float, usm_inst['pose'][3]))],
                                                dtype=np.float64)


    joint_values = np.asarray([list(map(float, usm_inst['articulation'][0])),
                             list(map(float, usm_inst['articulation'][1])),
                             list(map(float, usm_inst['articulation'][2]))],
                            dtype=np.float64)

    joint_values[-1] = 2 * joint_values[-1]

    # joint_values = list(map(float, usm_inst['articulation'][0]))
    # joint_values[-1] = 2 * joint_values[-1]



    # usm_inst = data['usm_2']

    # for key, value in data:
    #     if key is 'fame_index':  # 'name' is the key we wish to get the value from
    #         print(value)  # print its value
    # for p in data['fame_index']:
    #     print('usm-1: ' + p['usm-1'])

    for p in data:
        print(p, data[p])

#
#
# # mars_data_dir = '/opt/isi/mars_fs/data'
# mars_data_dir = '/data'
# json_file = open('data/data.json')
#
# data = json.load(json_file)
# # data = json.load(open('/home/max/platform/repos/marker-tracking/data/seq2/frame_gt10150.json'))
# usm_camera = data['usm_1']
# usm_inst = data['usm_2']
#
# # davinci = pysilico.PyDaVinci(mars_data_dir, "NONE", "THIRTY_SCOPE", "FENESTRATED_BIPOLAR_FORCEPS", "NONE")
# # davinci.AddShaderRenderer()
#
# instrument_to_camera_transform = np.asarray([list(map(float, usm_inst['instrument_to_camera_transform'][0])),
#                                              list(map(float, usm_inst['instrument_to_camera_transform'][1])),
#                                              list(map(float, usm_inst['instrument_to_camera_transform'][2])),
#                                              list(map(float, usm_inst['instrument_to_camera_transform'][3]))], dtype=np.float64)
#
# joint_values = list(map(float, usm_inst['instrument_joint_values']))
# joint_values[-1] = 2*joint_values[-1]
# davinci.SetInstrumentTransform(2, pysilico.Matrix(instrument_to_camera_transform), joint_values)


davinci.RenderInstruments()
(component_image_, camera_vertex_image_, model_vertex_image_) = davinci.GetShaderFramebuffers()

component_image = np.array(component_image_)

component_image = np.flipud(component_image)

imageio.imsave('/tmp/test_projection.png', component_image)