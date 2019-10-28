import json
import torch
import argparse
import os
import numpy as np
# import pysilico
import json
import numpy as np
import imageio

with open('dictSave/testsave1000_.json') as json_file:
    data = json.load(json_file)
data_len = len(data)
AllDataPoint = data
currentFrameId = 0  # contain the frame number to pick in the set
span = AllDataPoint[0]['Span']  # jump between frames to see the tool moving
number_frame = 0  # diplayed frames count, image count
TotNumbOfImage = len(data)  # each x frame will be picked, ideally 1000 for the ground truth database


