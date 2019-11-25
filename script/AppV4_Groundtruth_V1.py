from tkinter import *
from PIL import ImageTk, Image, ImageDraw
from tkinter import filedialog
import os
import json
import math
import tqdm
from scipy.misc import imsave
from functools import partial
import tkinter as tk
import cv2
import os
from skimage.io import imread, imsave
import glob
import imageio
import json
import argparse
import numpy as np
import neural_renderer as nr
import torch
from utils_functions.camera_settings import camera_setttings
import matplotlib.pyplot as plt

##### PARAMETERS GO HERE ###########

span = 1
TotNumbOfImage =19226

shaft_diameter = 8.25*1e-3
# plt.close("all")
# pathfile = 'framestest'
pathfile = 'framesLeft'
c_x = 590
c_y = 508

f_x = 1067
f_y = 1067



camera_calibration = np.zeros((4,4))
camera_calibration[0,0] = f_x
camera_calibration[1,1] = f_y
camera_calibration[0,2] = c_x
camera_calibration[1,2] = c_y
camera_calibration[2,2] = 1

current_dir = os.path.dirname(os.path.realpath(__file__))
sil_dir = os.path.join(current_dir, 'SilOutput')

parser = argparse.ArgumentParser()
parser.add_argument('-or', '--filename_output', type=str,
                    default=os.path.join(sil_dir, 'ResultSilhouette_{}.gif'.format('wtf')))
parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()

class CommandWindow:
    def __init__(self, master):
        self.pathfile = pathfile
        self.master = master
        self.frame = tk.Frame(self.master)
        self.interface_creation()


        # self.TotNumbOfImage = 4  # each x frame will be picked, ideally 1000 for the ground truth database
        # self.createDict()

    def goToImageNumber(self):
        frame2reach = np.int(self.entryfield.get()) #image to reach in the database, correspond of the position of the immage and NOT the image name (ID)
        self.currentFrameId = self.AllDataPoint[0]['FrameId']
        self.number_frame = frame2reach
        print('image to reach is the {}th image with id {}'.format(self.number_frame, self.currentFrameId))
        self.print_Status()
        self.new_window()


    def interface_creation(self):

        #frame creation
        self.buttonOpen = tk.Button(self.frame, text = 'Open Image', width = 20, command = self.new_window)
        self.buttonOpen.grid(row=0, column=0)
        self.buttonClose = tk.Button(self.frame, text = 'close', width = 20, command = self.close_image)
        self.buttonClose.grid(row=1, column=0)
        self.entryfield= tk.Entry(self.frame)
        self.entryfield.grid(row=1, column=5)
        self.entrybutton =  tk.Button(self.frame, text = 'reach', width = 10, command = self.goToImageNumber)
        self.entrybutton.grid(row=1, column=6)
        self.buttonNext = tk.Button(self.frame, text = 'Next', width = 20, command = self.next_frame)
        self.buttonNext.grid(row=2, column=0)
        self.buttonPrev = tk.Button(self.frame, text = 'Previous', width = 20, command = self.prev_frame)
        self.buttonPrev.grid(row=3, column=0)
        self.buttonLoadDict = tk.Button(self.frame, text = 'Load Dict', width = 10, command = self.load_dict)
        self.buttonLoadDict.grid(row=0, column=6)
        self.buttonSaveAll = tk.Button(self.frame, text = 'save all', width = 10, command = self.saveDict)
        self.buttonSaveAll.grid(row=11, column=6)

        # self.frame.bind('<Motion>', self.motion)


        #set default value
        self.currentFrameId = 0 #contain the frame number to pick in the set
        self.span = span # jump between frames to see the tool moving
        self.number_frame = 0 # diplayed frames count, image count
        self.TotNumbOfImage = TotNumbOfImage # each x frame will be picked, ideally 1000 for the ground truth database
        self.drawOK = True #will be disable if we create ground truth databse
        self.im2reach = 0


        self.python_green = "green"
        self.python_red = "red"
        self.python_blue = "blue"

        self.val_x =StringVar()
        self.val_y=StringVar()
        self.LabelSave = Label(self.frame, text='Dictionnary saved')

        self.entriesVarX = []
        self.entriesVarY = []
        self.entriesX = []
        self.entriesY = []
        self.updateEntryX = []
        self.updateEntryY = []
        self.buttonEdit = [] # tk.Button(self.frame, text='edit', width=10, command= lambda: (self.clearPose('R','1',token=1)))
        self.buttonVal = [] # tk.Button(self.frame, text='val', width=10, command=self.valPose)
        self.buttonPlotLine = []
        self.DictNameTable = \
                ["Redx1",
                "Redy1",
                "Redx2",
                "Redy2",
                "Greenx1",
                "Greeny1",
                "Greenx2",
                "Greeny2",
                "Bluex1",
                "Bluey1",
                "Bluex2",
                "Bluey2"]
        self.createDict()
        self.ColorTable = [
            self.python_red,
            self.python_red,
            self.python_green,
            self.python_green,
            self.python_blue,
            self.python_blue
        ]

        # for n in range(self.TotNumbOfImage):

        #create every interface button for each point in a loop
        for n in range(6):
            self.entriesVarX.append(StringVar())
            self.entriesVarY.append(StringVar())
            self.entriesVarX[n].set(n)
            self.entriesVarY[n].set(n)
            self.updateEntryX.append(False)
            self.updateEntryY.append(False)
            self.buttonEdit.append(tk.Button(self.frame, text='edit', width=10,command=partial(self.clearPose, n)))
            if n == 0: #only create those stuff ones
                self.buttonVal.append(tk.Button(self.frame, text='AutoAdv', width=10, command=partial(self.AutoAdv, n)))
                self.buttonVal[n].grid(row=5 + n, column=6)
                self.buttonPlotLine.append(tk.Button(self.frame, text='Create GT', width=10, command=partial(self.GTcreation, n))) #befor elf.ComputePointAngle
                self.buttonPlotLine[n].grid(row=5 + 3, column=6)

            print(n)
            # create entries list
            self.entriesX.append(Label(self.frame, width = 5, textvariable = self.entriesVarX[n]))
            self.entriesY.append(Label(self.frame, width = 5, textvariable = self.entriesVarY[n]))
            # # grid layout the entries
            self.entriesX[n].grid(row=5+n, column=1)
            self.entriesY[n].grid(row=5+n, column=4)
            self.buttonEdit[n].grid(row=5+n, column=5)


        #write every label separately for each color in the GUI
        # #Red dot 1 ----------------------------------------------------------------------------
        self.Label_Rx1 = Label(self.frame, text='0)Red_1 x')
        self.Label_Rx1.grid(row=5, column=0)

        self.Label_Ry1 = Label(self.frame, text='Red_1 y')
        self.Label_Ry1.grid(row=5, column=3)


        #Red dot 2 ----------------------------------------------------------------------------
        self.Label_Rx2 = Label(self.frame, text='1)Red_2 x')
        self.Label_Rx2 .grid(row=6, column=0)
        self.Label_Ry2 = Label(self.frame, text='Red_2 y')
        self.Label_Ry2 .grid(row=6, column=3)
        # Green dot 1 ----------------------------------------------------------------------------
        self.Label_Gx1 = Label(self.frame, text='2)Green_1 x')
        self.Label_Gx1.grid(row=7, column=0)
        self.Label_Gy1 = Label(self.frame, text='Green_1 y')
        self.Label_Gy1.grid(row=7, column=3)
        # Green dot 2 ----------------------------------------------------------------------------
        self.Label_Gx2 = Label(self.frame, text='3)Green_2 x')
        self.Label_Gx2.grid(row=8, column=0)
        self.Label_Gy2 = Label(self.frame, text='Green_2 y')
        self.Label_Gy2.grid(row=8, column=3)
        # Blue dot 1 ----------------------------------------------------------------------------
        self.Label_Bx1 = Label(self.frame, text='4)Blue_1 x')

        self.Label_Bx1.grid(row=9, column=0)
        self.Label_By1 = Label(self.frame, text='Blue_1 y')
        self.Label_By1.grid(row=9, column=3)
        # Blue dot 2 ----------------------------------------------------------------------------
        self.Label_Bx2 = Label(self.frame, text='5)Blue_2 x')
        self.Label_Bx2.grid(row=10, column=0)
        self.Label_By2 = Label(self.frame, text='Blue_2 y')
        self.Label_By2.grid(row=10, column=3)


        self.app_created = False #true if child is created
        self.autoAdvanced = False
        self.AllowDrawLine = False
        self.currentToken = 0
        self.print_Status()
        self.frame.pack()


    def print_Status(self):
        self.v = StringVar()
        self.v.set("image {} , id={}/{}, span {}".format(self.number_frame+1 , self.currentFrameId, self.TotNumbOfImage, self.span))
        self.LabelImageProcess = Label(self.frame, textvariable=self.v)
        self.LabelImageProcess.grid(row=0, column=4)

    def load_dict(self):
        self.LoadedDict = filedialog.askopenfilename(initialdir="/home/pierrec/Documents/Master_Thesis/ChirurgicalCADModel_RealDatabaset/script/dictSave", title="Select file",
                                                   filetypes=(("json files", "*.json"), ("all files", "*.*")))
        if (len(self.LoadedDict) != 0): #if filename exist is not eempty
            with open('{}'.format(self.LoadedDict)) as json_file:
                data = json.load(json_file) #load data
                self.AllDataPoint=data
                if self.app_created: #close current by saving
                    self.close_image()

                self.currentFrameId = 0  # contain the frame number to pick in the set
                self.span = self.AllDataPoint[0]['Span']  # jump between frames to see the tool moving
                self.number_frame = 0  # diplayed frames count, image count
                self.TotNumbOfImage = len(data)  # each x frame will be picked, ideally 1000 for the ground truth database
                print('loaded dictionary contains {} positions'.format(self.TotNumbOfImage))
                self.print_Status()

    def saveDict(self):
        self.Savefilename = filedialog.asksaveasfilename(initialdir="/home/pierrec/Documents/Master_Thesis/ChirurgicalCADModel_RealDatabaset/script/dictSave", title="Select file",
                                                     filetypes=(("json files", "*.json"), ("all files", "*.*")))
        print(self.Savefilename)
        with open('{}'.format(self.Savefilename), 'w') as fp:
            json.dump(self.AllDataPoint, fp)

        self.LabelSave.grid(row=11, column=0)

    def new_window(self):
        #plot color point


        self.newWindow = tk.Toplevel(self.master)
        self.app = Child_window(self.newWindow, ImageId=self.currentFrameId)
        self.app_created = True
        self.app.canvas.bind('<Motion>', self.motion_all)

        for i in range(6):
            self.entriesVarX[i].set(self.AllDataPoint[self.number_frame][self.DictNameTable[i * 2]])
            self.entriesVarY[i].set(self.AllDataPoint[self.number_frame][self.DictNameTable[i * 2 + 1]])
            self.updateEntryX[i] = True
            self.updateEntryY[i] = True

        # self.autoAdvanced = True
        self.currentToken = 0  # auto advanced starts at first position
        self.First = True
        self.currentCanvaPoint()
        self.AllowDrawLine = True

        self.ComputePointAngle()


    def close_image(self):
        self.app.close_windows()
        self.app_created = False
        self.First = True


    def next_frame(self):
        # plot color point


        if self.app_created:
            self.updatePose = True
            if (self.number_frame < self.TotNumbOfImage-1): #if we still have picture to display
                if (self.currentFrameId + self.span <= 19226-1): #if the next picture is with the total image frame
                    self.currentFrameId = self.currentFrameId+self.span
                    self.number_frame = self.number_frame + 1
                else:
                    self.currentFrameId = self.currentFrameId


            self.close_image()
            self.new_window()
            print(self.number_frame)
            for i in range(6):
                self.entriesVarX[i].set(self.AllDataPoint[self.number_frame][self.DictNameTable[i * 2]] )
                self.entriesVarY[i].set(self.AllDataPoint[self.number_frame][self.DictNameTable[i * 2 + 1]])


            self.currentToken = 0 #auto advanced starts at first position
            self.First = True
            self.currentCanvaPoint()
            self.AllowDrawLine = True

            # self.showColorPoint()
            self.print_Status()

    def prev_frame(self):
        # plot color point

        if self.app_created:
            #selection of the frame in the database
            self.updatePose = True
            if (self.currentFrameId-self.span >= 0):
                self.currentFrameId = self.currentFrameId-self.span
                self.number_frame = self.number_frame - 1
            else:
                self.currentFrameId = self.currentFrameId
            self.close_image()
            self.new_window()

            #update field with already known coordinate stored in the dictionnary
            for i in range(6):
                self.entriesVarX[i].set(self.AllDataPoint[self.number_frame][self.DictNameTable[i * 2]] )
                self.entriesVarY[i].set(self.AllDataPoint[self.number_frame][self.DictNameTable[i * 2 + 1]])

            self.currentToken = 0 #auto advanced starts at first position
            self.First = True
            self.currentCanvaPoint() #create the data base of the color point with .First = True to create and draw
            self.AllowDrawLine = True


            self.print_Status()

    def motion_all(self,event):

        if self.app_created:
            if self.updateEntryX[self.currentToken]: #update allowed?
                if self.app.clk:

                    #update field
                    self.entriesVarX[self.currentToken].set(self.app.x)
                    self.entriesVarY[self.currentToken].set(self.app.y)

                    # #update dictionnary axis
                    self.AllDataPoint[self.number_frame][self.DictNameTable[self.currentToken*2]] = self.app.x
                    self.AllDataPoint[self.number_frame][self.DictNameTable[self.currentToken*2+1]] = self.app.y

                    # #update dictionnary frame number
                    self.AllDataPoint[self.number_frame]['FrameId'] = self.currentFrameId


                    self.currentCanvaPoint() #update the data base of the color point .First = False to update and draw
                    self.AllowDrawLine = True


                    if self.autoAdvanced:
                        if self.currentToken < 5:
                            self.currentToken = self.currentToken+1
                            print(self.currentToken)
                            self.updateEntryX[self.currentToken] = True
                            self.updateEntryY[self.currentToken] = True
                            self.app.clk = False
                        else:
                            self.currentToken = 0
                            self.updateEntryX[self.currentToken] = True
                            self.updateEntryY[self.currentToken] = True
                            self.app.clk = False
                            self.next_frame()


                    self.app.clk = False

    def clearPose(self, n):
        print(n)
        self.entriesVarX[n].set(0)
        self.entriesVarY[n].set(0)
        self.currentToken = n
        self.AllDataPoint[self.number_frame][self.DictNameTable[self.currentToken * 2]] = 0
        self.AllDataPoint[self.number_frame][self.DictNameTable[self.currentToken * 2 + 1]] = 0
        self.updateEntryX[n] = True
        self.updateEntryY[n] = True

        # for i in range(6):
        # self.First=True
        # self.currentCanvaPoint()
        # self.autoAdvanced = False
        # self.Label_Rx1 = Label(self.frame, text='  off  ')
        # self.Label_Rx1.grid(row=6, column=6)
        self.app.canvas.delete(self.TablecurrentCanvaPoint[n])


    def AutoAdv(self, n):
        self.autoAdvanced = not self.autoAdvanced
        print('auto advanced is {}'.format(self.autoAdvanced))

        if self.autoAdvanced:
            self.Label_Rx1 = Label(self.frame, text='  on  ')
            self.Label_Rx1.grid(row=6, column=6)
        else:
            self.Label_Rx1 = Label(self.frame, text='  off  ')
            self.Label_Rx1.grid(row=6, column=6)



    def showColorPoint(self):
        for i in range(6):


            x = self.AllDataPoint[self.number_frame][self.DictNameTable[i * 2]]
            y = self.AllDataPoint[self.number_frame][self.DictNameTable[i * 2 + 1]]

            size = 4
            x1, y1 = (x - size), (y - size)
            x2, y2 = (x + size), (y + size)
            if x !=0 and y !=0 :
                self.app.canvas.create_oval(x1, y1, x2, y2, fill=self.ColorTable[i])
            # else

    def clearPoint(self):
        self.app.canvas.delete(all)
        # for i in range(6):

    def ComputePointAngle(self,n=0):
        self.point2pointAngle = []
        self.PointTableWithColor = []
        self.AllPointWithColor = []
        self.LineColorCombination = []
        self.LineNumberCombination = []
        self.Angles = []
        self.AngleThreshold = 4


        for i in range(6):
            x = self.AllDataPoint[self.number_frame][self.DictNameTable[i * 2]]
            # self.currentValue.append(x)
            y = self.AllDataPoint[self.number_frame][self.DictNameTable[i * 2 + 1]]

            if x !=0 and y !=0:
                if i == 0 or i==1:
                    if i%2 == 0:
                        self.PointTableWithColor.append([x,y,'R',1])
                    else:
                        self.PointTableWithColor.append([x,y,'R',2])
                if i == 2 or i==3:
                    if i%2 == 0:
                        self.PointTableWithColor.append([x,y,'G',1])
                    else:
                        self.PointTableWithColor.append([x,y,'G',2])
                if i == 4 or i==5:
                    if i%2 == 0:
                        self.PointTableWithColor.append([x,y,'B',1])
                    else:
                        self.PointTableWithColor.append([x,y,'B',2])

        # print(len(self.PointTableWithColor))
        for i in range(len(self.PointTableWithColor)):
            point_start = self.PointTableWithColor[i][0:2]
            color_start = self.PointTableWithColor[i][2]
            color_number_start = self.PointTableWithColor[i][3]
            # print('starting color is now {} {} at {}'.format(color_start,color_number_start, point_start))

            second_point_candidate = []
            for j in range(len(self.PointTableWithColor)): #create the [4,3] list without the color start
                if self.PointTableWithColor[j][2] != color_start: #check the color of each row
                    # print( self.PointTableWithColor[i])
                    candidate = self.PointTableWithColor[j]
                    second_point_candidate.append(candidate)
            # print(len(point_candidate))

            #in the new 4x3 candidate matrix, pick the first one and compare ange with next, store if same and then iterate in the 4x3table
            for k in range(len(second_point_candidate)):
                # print(second_point_candidate[k])
                color_second = second_point_candidate[k][2]
                color_number_second =  second_point_candidate[k][3]
                third_point_candidate = []
                #now build the 2x2 list
                for l in range(len(second_point_candidate)):


                    if second_point_candidate[l][2] != color_second:

                        third_candidate = second_point_candidate[l]
                        color_third = third_candidate[2]

                        third_point_candidate.append(third_candidate)

                # #go through
                # for n in range(len(second_point_candidate))
                alpha1 = math.degrees(math.atan2(point_start[1]-second_point_candidate[k][1], point_start[0]-second_point_candidate[k][0]))
                # print('alpha1 is {} between {}{} and {}{}'.format(alpha1,color_start,color_number_start,color_second,color_number_second))
                #comupute Alpha 2 for the remaining color-
                for m in range(len(third_point_candidate)):
                    color_number_third = third_point_candidate[m][3]
                    alpha2_cand =  math.degrees(math.atan2(second_point_candidate[k][1]-third_point_candidate[m][1],second_point_candidate[k][0]- third_point_candidate[m][0]))
                    # print('alpha2 is {} between {}{} and {}{}'.format(alpha2_cand, color_second,color_number_second,color_third,color_number_third))
                    # print('alpha2 is {}'.format(alpha2_cand))
                    if alpha2_cand > alpha1-self.AngleThreshold and alpha2_cand < alpha1 + self.AngleThreshold:
                        code_color = [color_start,color_second,color_third]
                        code_number = [color_number_start,color_number_second,color_number_third]
                        # print(code_color)
                        # print(code_number)
                        self.LineColorCombination.append(code_color)
                        self.LineNumberCombination.append(code_number)

                        # self.drawCanvaLine(code_color,code_number)
        self.drawCanvaAllLine()

    def drawCanvaAllLine(self):
        self.TablecurrentAllCanvaLine = []
        firstpoint = []
        secondpoint = []
        thirdpoint = []
        #go through all line combination of the table
        for i in range(len(self.LineColorCombination)):
            line_start_point = self.LineColorCombination[i][0]
            line_start_point_number = self.LineNumberCombination[i][0]
            line_middle_point = self.LineColorCombination[i][1]
            line_middle_point_number = self.LineNumberCombination[i][1]
            line_stop_point = self.LineColorCombination[i][2]
            line_stop_point_number = self.LineNumberCombination[i][2]

            #find the starting point coordinates in the table
            for j in range(len(self.PointTableWithColor)):
                if line_start_point in self.PointTableWithColor[j]:
                   if line_start_point_number in self.PointTableWithColor[j]: #if the line found is the correct one
                    x1 = self.PointTableWithColor[j][0]
                    y1 = self.PointTableWithColor[j][1]

            # find the middle point coordinates in the table
            for k in range(len(self.PointTableWithColor)):
                if line_middle_point in self.PointTableWithColor[k]:
                    if line_middle_point_number in self.PointTableWithColor[k]:
                        x1_2 = self.PointTableWithColor[k][0]
                        y1_2 = self.PointTableWithColor[k][1]

            #find the ending point coordinates in the table
            for l in range(len(self.PointTableWithColor)):
                if line_stop_point in self.PointTableWithColor[l]:
                   if line_stop_point_number in self.PointTableWithColor[l]:
                    x2= self.PointTableWithColor[l][0]
                    y2 = self.PointTableWithColor[l][1]

            distP1_P12 = math.sqrt((x1 - x1_2) ** 2 + (y1 - y1_2) ** 2)
            distP2_P12 = math.sqrt((x2 - x1_2) ** 2 + (y2 - y1_2) ** 2)

            if distP2_P12<distP1_P12:
                #flip the order
                tempx = x1
                tempy = y1
                x1 = x2
                y1 = y2
                x2 = tempx
                y2 = tempy
                #swap in the table
                temp_color = self.LineColorCombination[i][0]
                temp_number =  self.LineNumberCombination[i][0]
                self.LineColorCombination[i][0] = self.LineColorCombination[i][2]
                self.LineNumberCombination[i][0] = self.LineNumberCombination[i][2]
                self.LineColorCombination[i][2] = temp_color
                self.LineNumberCombination[i][2] = temp_number


            firstpoint.append([x1,y1])
            secondpoint.append([x1_2,y1_2])
            thirdpoint.append([x2,y2])


            text_dist = 4


        #remove duplicate in every table
        self.no_dupes_NumberCombination = [x for n, x in enumerate(self.LineNumberCombination) if x not in self.LineNumberCombination[:n]]
        self.no_dupes_ColorCombination = [x for n, x in enumerate(self.LineColorCombination) if x not in self.LineColorCombination[:n]]
        self.no_dupes_FirstPointCoord =  [x for n, x in enumerate(firstpoint) if x not in firstpoint[:n]]
        self.no_dupes_SecondPointCoord = [x for n, x in enumerate(secondpoint) if x not in secondpoint[:n]]
        self.no_dupes_ThirdPointCoord = [x for n, x in enumerate(thirdpoint) if x not in thirdpoint[:n]]
        # for item in range(len(self.LineNumberCombination)):

        if self.drawOK:
            for i in range(len(self.no_dupes_NumberCombination)):
                self.app.canvas.create_line(self.no_dupes_FirstPointCoord [i][0], self.no_dupes_FirstPointCoord [i][1], self.no_dupes_ThirdPointCoord [i][0], self.no_dupes_ThirdPointCoord [i][1], fill='red')
                self.app.canvas.create_text(self.no_dupes_FirstPointCoord [i][0] + text_dist, self.no_dupes_FirstPointCoord [i][1] + text_dist, anchor='nw', text='1', fill='red')
                self.app.canvas.create_text(self.no_dupes_SecondPointCoord [i][0] + text_dist, self.no_dupes_SecondPointCoord [i][1] + text_dist, anchor='nw', text='2', fill='red')
                self.app.canvas.create_text(self.no_dupes_ThirdPointCoord [i][0]+ text_dist, self.no_dupes_ThirdPointCoord [i][1] + text_dist, anchor='nw', text='3', fill='red')


        #search for the transform given 6 points points
        state = self.compute_initial_transform()

        if state: #if a transform T_m as been found, state is true
            self.renderingGivenTm()



    def currentCanvaPoint(self):
        if self.First:
            self.TablecurrentCanvaPoint = []
        # self.previousValue =
        for i in range(6):
            x = self.AllDataPoint[self.number_frame][self.DictNameTable[i * 2]]
            # self.currentValue.append(x)
            y = self.AllDataPoint[self.number_frame][self.DictNameTable[i * 2 + 1]]
            # self.currentValue.append(y)
            size = 4
            x1, y1 = (x - size), (y - size)
            x2, y2 = (x + size), (y + size)

            if self.First: #if first, build the list

                if x != 0 and y != 0:
                #creation of the list of 6 point of the current canva
                    # item2delete = self.TablecurrentCanvaPoint[i]
                    self.TablecurrentCanvaPoint.append(self.app.canvas.create_oval(x1, y1, x2, y2, fill=self.ColorTable[i]))
                    # self.app.canvas.delete(item2delete)
                else:
                    self.TablecurrentCanvaPoint.append(0)


            else:
                if x != 0 and y != 0:
                    self.app.canvas.delete(self.TablecurrentCanvaPoint[i])
                    self.TablecurrentCanvaPoint[i] = self.app.canvas.create_oval(x1, y1, x2, y2, fill=self.ColorTable[i])
                    # self.app.canvas.delete(item2delete)
                else:
                    self.TablecurrentCanvaPoint[i]=0


        self.First = False


    def createDict(self): #creation of the dictionnary, one full set of point for each frame of the video

        self.AllDataPoint = []

        for i in range(0, self.TotNumbOfImage):  # creation of the list of dictionnary
            OneFrameDict = {
                "ImageNo": i,
                "Span": self.span,
                "FrameId":0,
                self.DictNameTable[0]: 0,
                self.DictNameTable[1]: 0,
                self.DictNameTable[2]: 0,
                self.DictNameTable[3]: 0,
                self.DictNameTable[4]: 0,
                self.DictNameTable[5]: 0,
                self.DictNameTable[6]: 0,
                self.DictNameTable[7]: 0,
                self.DictNameTable[8]: 0,
                self.DictNameTable[9]: 0,
                self.DictNameTable[10]: 0,
                self.DictNameTable[11]: 0,
            }
            self.AllDataPoint.append(OneFrameDict)

        # print('dictionnary created with {} elements'.format(self.TotNumbOfImage))

    def rotate_point_around_shaft(self,point, a):
        r_m = np.zeros((4, 4))
        a = np.deg2rad(a)
        r_m[0, 0] = np.cos(a)
        r_m[0, 1] = np.sin(a)
        r_m[1, 0] = -np.sin(a)
        r_m[1, 1] = np.cos(a)
        r_m[2, 2] = 1
        r_m[3, 3] = 1


        return np.matmul(r_m, point)


    def compute_initial_transform(self):

        camera_points = np.empty((0, 2))
        model_points = np.empty((0, 3))
        p_1 = [0, shaft_diameter / 2, -1.5 * 1e-2, 1]
        p_2 = [0, shaft_diameter / 2, -2.5 * 1e-2, 1]
        p_3 = [0, shaft_diameter / 2, -4.0 * 1e-2, 1]
        for l in range(len(self.no_dupes_ColorCombination)):
            p = self.no_dupes_ColorCombination[l]
            #test
            # p[0] ='G'
            # p[1] ='R'
            # p[2] ='B'
            found = False

            if (p[0] == 'R'  and p[1] == 'G' and p[2] == 'B'):  # PGB
                rot = 0
                found = True
            elif (p[0] == 'B' and p[1] == 'R'  and p[2] == 'G'):  # BPG
                rot = 60
                found = True
            elif (p[0] == 'G' and p[1] == 'B' and p[2] == 'R' ):  # GBP
                rot = 120
                found = True
            elif (p[0] == 'R'  and p[1] == 'B' and p[2] == 'G'):  # PBG
                rot = 180
                found = True
            elif (p[0] == 'B' and p[1] == 'G' and p[2] == 'R' ):  # BGP
                rot = 240
                found = True
            elif (p[0] == 'G' and p[1] == 'R'  and p[2] == 'B'):  # GPB
                rot = 300
                found = True
            elif (p[0] == p[1] and p[1] == p[2]):
                found = False

            # print('rotation of {}{}{} is {} degree'.format(p[0], p[1], p[2], rot))

            #first point, closest to origin in the camera space
            camera_points = np.vstack((camera_points, np.expand_dims(self.no_dupes_FirstPointCoord[l], axis=0)))
            #second point
            camera_points = np.vstack((camera_points, np.expand_dims(self.no_dupes_SecondPointCoord[l], axis=0)))
            #third point, furthest to origin in the camera space
            camera_points = np.vstack((camera_points, np.expand_dims(self.no_dupes_ThirdPointCoord[l], axis=0)))
            #equivalence of the first point in the 3d model space, and the second and third point converted into model space
            model_points = np.vstack((model_points, self.rotate_point_around_shaft(p_1, rot)[0:3]))
            model_points = np.vstack((model_points, self.rotate_point_around_shaft(p_2, rot)[0:3]))
            model_points = np.vstack((model_points, self.rotate_point_around_shaft(p_3, rot)[0:3]))


        if (camera_points.shape[0] > 3):


            # for i in range(camera_points.shape[0]):
            #     print('model point {} projects to camera pixel {}'.format(model_points[i], camera_points[i]))

            if (camera_points.shape[0] > 3):
                expModel_points = np.expand_dims(model_points, axis=2)
                expCamera_points = np.expand_dims(camera_points, axis=2)
                IntCam = camera_calibration[0:3, 0:3]

                retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=model_points,
                                                                 imagePoints=camera_points,
                                                                 cameraMatrix=camera_calibration[0:3, 0:3],
                                                                 distCoeffs=None)



            R = cv2.Rodrigues(rvec)[0]
            # R = self.rotate_correction(R, 0)
            R2 = cv2.Rodrigues(rvec)
            T_m = np.zeros((4, 4))
            T_m[0:3, 0:3] = R
            T_m[0:3, 3] = np.squeeze(tvec)
            T_m[3, 3] = 1

            self.T_m = T_m #brings points from the model coordinate system to the camera coordinate system


            # print('found an updated transform')
            # print(self.T_m)

            #pinhole camera to project back the 3d point into 2d space
            pinhole_point1 = np.empty((0, 3))
            for i in range(len(model_points)):
                pointcoord =  model_points[i,:]
                pointRot = np.matmul(R,pointcoord)
                pointTrans = pointRot + tvec.T
                pinhole_point1 =  np.vstack((pinhole_point1, np.expand_dims(pointTrans[0], axis=0)))

            self.pinhole_point1 = pinhole_point1
            f = f_x
            # X_recov = -(f/)
            pinhole_point2 = np.empty((0, 2))
            for i in range(len(model_points)):
                px = (f/pinhole_point1[i,2])*pinhole_point1[i,0] +c_x
                py = (f /pinhole_point1[i, 2]) * pinhole_point1[i, 1] + c_y
                pinhole_point2 = np.vstack((pinhole_point2, np.expand_dims([px,py], axis=0)))

            # plt.scatter(pinhole_point2[:,0], pinhole_point2[:,1])
            # plt.show()
            self.pinhole_point2 = pinhole_point2
            return True

        else:

            print('failed to find an updated transform. not enough matched points. we only found {}'.format(
                camera_points.shape[0]))

            return False

    def matrix2angle(self,instrument_to_camera_transform):

        #angle and translation vector extraction from transformation matrix
        Extracted_theta3_rad = math.atan2(instrument_to_camera_transform[1, 0], instrument_to_camera_transform[0, 0])
        C_2 = math.sqrt(instrument_to_camera_transform[2, 1] * instrument_to_camera_transform[2, 1] +
                     instrument_to_camera_transform[2, 2] * instrument_to_camera_transform[2, 2])
        Extracted_theta2_rad = math.atan2(-instrument_to_camera_transform[2, 0], C_2)
        Extracted_theta1_rad = math.atan2(instrument_to_camera_transform[2, 1], instrument_to_camera_transform[2, 2])

        Extracted_X = instrument_to_camera_transform[0, 3]
        Extracted_Y = instrument_to_camera_transform[1, 3]
        Extracted_Z = instrument_to_camera_transform[2, 3]

        Extracted_theta1_deg = np.degrees(Extracted_theta1_rad)
        Extracted_theta2_deg = np.degrees(Extracted_theta2_rad)
        Extracted_theta3_deg = np.degrees(Extracted_theta3_rad)

        return Extracted_X, Extracted_Y, Extracted_Z, Extracted_theta1_deg, Extracted_theta2_deg, Extracted_theta3_deg


    def renderingGivenTm(self):
        # print('rendering the 3D cad tool')

        instrument_to_camera_transform = self.T_m

        Extracted_X, Extracted_Y, Extracted_Z, Extracted_theta1_deg, Extracted_theta2_deg, Extracted_theta3_deg = self.matrix2angle(self.T_m)


        # define transfomration parameter from json file
        alpha =Extracted_theta1_deg+90 #adapt from openCV to renderer axis system
        beta = Extracted_theta2_deg
        gamma =Extracted_theta3_deg
        x = Extracted_X
        y = Extracted_Y
        z = Extracted_Z
        # print('parameter found are: ',x, y, z, alpha, beta, gamma)

        #renderer the 3D cad model
        vertices_1, faces_1, textures_1 = nr.load_obj("3D_objects/shaftshortOnly.obj", load_texture=True, normalization=False)  # , texture_size=4)
        vertices_1 = vertices_1[None, :, :]  # add dimension
        faces_1 = faces_1[None, :, :]  # add dimension
        textures_1 = textures_1[None, :, :]  # add dimension
        nb_vertices = vertices_1.shape[0]

        R = np.array([np.radians(alpha), np.radians(beta), np.radians(gamma)])  # angle in degree
        t = np.array([x, y, z])  # translation in meter

        # define transformation by transformation matrix

        self.Rt = np.concatenate((R, t), axis=None).astype(np.float16)  # create one array of parameter in radian, this arraz will be saved in .npy file

        cam = camera_setttings(R=R, t=t, PnPtm = self.T_m, PnPtmFlag = False, vert=nb_vertices, resolutionx=1280, resolutiony=1024,cx=c_x, cy=c_y, fx=f_x, fy=f_y) # degree angle will be converted  and stored in radian

        renderer = nr.Renderer(image_size=1280, camera_mode='projection', dist_coeffs=None,anti_aliasing=True, fill_back=True, perspective=False,
                               K=cam.K_vertices, R=cam.R_vertices, t=cam.t_vertices, near=0, background_color=[1, 1, 1],
                               # background is filled now with  value 0-1 instead of 0-255
                               # changed from 0-255 to 0-1
                               far=1, orig_size=1280,
                               light_intensity_ambient=1, light_intensity_directional=0.5, light_direction=[0, 1, 0],
                               light_color_ambient=[1, 1, 1], light_color_directional=[1, 1, 1])

        images_1 = renderer(vertices_1, faces_1, textures_1,
                            K=torch.cuda.FloatTensor(cam.K_vertices),
                            R=torch.cuda.FloatTensor(cam.R_vertices),
                            t=torch.cuda.FloatTensor(cam.t_vertices))  # [batch_size, RGB, image_size, image_size]

        image = images_1[0].detach().cpu().numpy()[0].transpose((1, 2, 0)) #float32 from 0-1
        image = (image*255).astype(np.uint8) #cast from float32 255.0 to 255 uint8

        self.image = image[0:1024,0:1280,:]

        #modifification done in the rasterize.py file for the  default far and near value  DEFAULT_FAR = 1 , DEFAULT_EPS = 1e-4
        sils_1 = renderer(vertices_1, faces_1, textures_1,
                          mode='silhouettes',
                          K=torch.cuda.FloatTensor(cam.K_vertices),
                          R=torch.cuda.FloatTensor(cam.R_vertices),
                          t=torch.cuda.FloatTensor(cam.t_vertices))  # [batch_size, RGB, image_size, image_size]

        sil = sils_1.detach().cpu().numpy().transpose((1, 2, 0))
        sil = np.squeeze((sil * 255)).astype(np.uint8) # change from float 0-1 [512,512,1] to uint8 0-255 [512,512]
        self.sil = sil[0:1024, 0:1280]


        #create window of the overlap of the tool and renderer
        backgroundImage = Image.open("{}/frameL{}.jpg".format(self.pathfile, self.currentFrameId))
        self.backgroundIm = np.array(backgroundImage)
        toolbck = backgroundImage.load()
        sil3d = self.sil[:, :, np.newaxis]
        renderim = np.concatenate((sil3d,sil3d,sil3d), axis=2)
        toolIm = Image.fromarray(np.uint8(renderim ))
        alpha = 0.2
        size = 10 #ellipse size
        out = Image.blend(backgroundImage,toolIm,alpha)
        draw = ImageDraw.Draw(out)
        for i in range(len(self.pinhole_point2)):
            px = self.pinhole_point2[i,0]
            py = self.pinhole_point2[i,1]
            draw.ellipse([px- size/2,py- size/2,px+ size/2, py+ size/2], fill='blue')
        # draw.show()
        draw.ellipse([c_x- size/2, c_y- size/2, c_x + size/2, c_y + size/2], fill='red')
        self.out = np.array(out)


        # out.show()
        #
        # fig = plt.figure()
        # fig.add_subplot(2, 1, 1)
        # plt.imshow(self.image)
        # # imageio.imwrite("3D_objects/{}_ref.png".format(file_name_extension), image)
        #
        # fig.add_subplot(2, 1, 2)
        # plt.imshow(self.sil, cmap='gray')
        # plt.show()


    def GTcreation(self,n=0): #this function will create the ground truth to train the neural network
        backgroundImage_database = []
        RGBshaft_database = []
        BWshaft_database = []
        params_database = []

        print('Ground truth creation')
        self.load_dict() #open the dictionnary to compute the ground truth
        self.NumberOfImageWith6Points = 0
        processcount = 0
        #for each position, control that we have 6 positions, if yes save them in a new directory and add alpha beta gamm x y z field
        loop = tqdm.tqdm(range(len(self.AllDataPoint)))
        # loop = tqdm.tqdm(range(0,20))
        for i in loop:
            if self.AllDataPoint[i]['Redx1'] != 0 and self.AllDataPoint[i]['Redx2'] != 0 and self.AllDataPoint[i]['Greenx1'] != 0 and self.AllDataPoint[i]['Greenx2'] != 0 and self.AllDataPoint[i]['Bluex1'] != 0 and self.AllDataPoint[i]['Bluex2'] != 0:
                self.NumberOfImageWith6Points = self.NumberOfImageWith6Points +1
                self.currentFrameId = self.AllDataPoint[i]['FrameId']
                self.number_frame = i
                self.drawOK = False #dont draw on the child windows cause it does not exist
                self.ComputePointAngle()


                backgroundImage_database.extend(self.backgroundIm)
                RGBshaft_database.extend(self.image)
                BWshaft_database.extend(self.sil)
                params_database.extend(self.Rt)

                # make gif
                imsave('/tmp/_tmp_%04d.png' % processcount, self.out)
                processcount = processcount + 1


        make_gif(args.filename_output)
        height, width, depth = np.shape(self.image) #1280x1024x3
        print('{}/{} have 6 points '.format(self.NumberOfImageWith6Points,self.TotNumbOfImage))

        # reshape for tensor format
        backgroundImage_database = np.reshape(backgroundImage_database, (self.NumberOfImageWith6Points,  height, width,  3))  # 3 channel rgb
        RGBshaft_database = np.reshape(RGBshaft_database, (self.NumberOfImageWith6Points,  height, width, 3))  # 3 channel rgb
        BWshaft_database = np.reshape(BWshaft_database, (self.NumberOfImageWith6Points,  height, width,))  # binary mask monochannel
        params_database = np.reshape(params_database, (self.NumberOfImageWith6Points, 6))  # array of 6 params, angle are stored in radian

        file_name_extension = '{}_images3'.format(self.NumberOfImageWith6Points)

        np.save('Npydatabase/endoscIm_{}.npy'.format(file_name_extension), backgroundImage_database)
        np.save('Npydatabase/RGBShaft_{}.npy'.format(file_name_extension), RGBshaft_database)
        np.save('Npydatabase/BWShaft_{}.npy'.format(file_name_extension), BWshaft_database)
        np.save('Npydatabase/params_{}.npy'.format(file_name_extension), params_database)
        print('Ground truth database saved')





#----------------------------------------------------------------------
class Child_window:
    def __init__(self, master, ImageId=0):
        self.pathfile = pathfile
        self.master = master
        self.x = 0
        self.y =0
        self.clk = False
        # master.minsize(width=1280, height=1024)
        self.frame = tk.Frame(self.master)
        self.master.title("Frame number {}".format(ImageId))
        imageinfo = Image.open("{}/frameL{}.jpg".format(self.pathfile,ImageId))
        self.img = ImageTk.PhotoImage(Image.open("{}/frameL{}.jpg".format(self.pathfile, ImageId)))
        # im2blend = self.img
        #     if 'self.image' in locals():
        #     self.blendImCad = im2blend.blend(self.img, self.image, 0.5)
        self.canvas = Canvas(self.master, width = imageinfo.size[0], height = imageinfo.size[1])
        # print( imageinfo.size[0], imageinfo.size[1]) #print canva size
        self.canvas.create_image(0,0, image=self.img, anchor="nw")
        self.canvas.pack()
        master.bind('<Button-1>', self.clicked)

    def close_windows(self):
        self.master.destroy()


    def clicked(self, event):
        self.x, self.y = event.x, event.y
        self.clk = True
        # print(self.x,self.y)


def main():
    root = tk.Tk()
    app = CommandWindow(root)
    # Get an instance of the MousePos object

    root.mainloop()

if __name__ == '__main__':
    main()

# https://stackoverflow.com/questions/16115378/tkinter-example-code-for-multiple-windows-why-wont-buttons-load-correctly