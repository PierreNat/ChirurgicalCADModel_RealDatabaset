from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import json
import math
from functools import partial
import tkinter as tk


class CommandWindow:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.interface_creation()


        # self.TotNumbOfImage = 4  # each x frame will be picked, ideally 1000 for the ground truth database
        # self.createDict()

    def interface_creation(self):

        #frame creation
        self.buttonOpen = tk.Button(self.frame, text = 'Open Image', width = 20, command = self.new_window)
        self.buttonOpen.grid(row=0, column=0)
        self.buttonClose = tk.Button(self.frame, text = 'close', width = 20, command = self.close_image)
        self.buttonClose.grid(row=1, column=0)
        self.buttonNext = tk.Button(self.frame, text = 'Next', width = 20, command = self.next_frame)
        self.buttonNext.grid(row=2, column=0)
        self.buttonPrev = tk.Button(self.frame, text = 'Previous', width = 20, command = self.prev_frame)
        self.buttonPrev.grid(row=3, column=0)
        self.buttonLoadDict = tk.Button(self.frame, text = 'Load Dict', width = 10, command = self.load_dict)
        self.buttonLoadDict.grid(row=0, column=6)
        self.buttonSaveAll = tk.Button(self.frame, text = 'save all', width = 10, command = self.saveDict)
        self.buttonSaveAll.grid(row=11, column=6)

        # self.frame.bind('<Motion>', self.motion)


        self.currentFrameId = 0 #contain the frame number to pick in the set
        self.span = 10 # jump between frames to see the tool moving
        self.number_frame = 0 # diplayed frames count, image count
        self.TotNumbOfImage = 1000 # each x frame will be picked, ideally 1000 for the ground truth database


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


        for n in range(6):
            self.entriesVarX.append(StringVar())
            self.entriesVarY.append(StringVar())
            self.entriesVarX[n].set(n)
            self.entriesVarY[n].set(n)
            self.updateEntryX.append(False)
            self.updateEntryY.append(False)
            self.buttonEdit.append(tk.Button(self.frame, text='edit', width=10,command=partial(self.clearPose, n)))
            if n == 0:
                self.buttonVal.append(tk.Button(self.frame, text='AutoAdv', width=10, command=partial(self.AutoAdv, n)))
                self.buttonVal[n].grid(row=5 + n, column=6)
                self.buttonPlotLine.append(tk.Button(self.frame, text='PlotLIne', width=10, command=partial(self.ComputePointAngle, n)))
                self.buttonPlotLine[n].grid(row=5 + 3, column=6)

            print(n)
            # create entries list
            self.entriesX.append(Label(self.frame, width = 5, textvariable = self.entriesVarX[n]))
            self.entriesY.append(Label(self.frame, width = 5, textvariable = self.entriesVarY[n]))
            # # grid layout the entries
            self.entriesX[n].grid(row=5+n, column=1)
            self.entriesY[n].grid(row=5+n, column=4)
            self.buttonEdit[n].grid(row=5+n, column=5)



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
        self.v.set("image {}/{}".format(self.number_frame+1, self.TotNumbOfImage))
        self.LabelImageProcess = Label(self.frame, textvariable=self.v)
        self.LabelImageProcess.grid(row=0, column=4)

    def load_dict(self):
        self.LoadedDict = filedialog.askopenfilename(initialdir="/home/pierrec/Documents/Master_Thesis/ChirurgicalCADModel_RealDatabaset/script/dictSave", title="Select file",
                                                   filetypes=(("json files", "*.json"), ("all files", "*.*")))
        if (len(self.LoadedDict) != 0):
            with open('{}'.format(self.LoadedDict)) as json_file:
                data = json.load(json_file)
                self.AllDataPoint=data
                if self.app_created: #close current by saving
                    self.close_image()

                self.currentFrameId = 0  # contain the frame number to pick in the set
                self.span = self.AllDataPoint[0]['Span']  # jump between frames to see the tool moving
                self.number_frame = 0  # diplayed frames count, image count
                self.TotNumbOfImage = len(data)  # each x frame will be picked, ideally 1000 for the ground truth database
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


        if self.app_created :
            self.updatePose = True
            if (self.number_frame < self.TotNumbOfImage-1): #if we still have picture to display
                if (self.currentFrameId + self.span <= 18000): #if the next picture is with the total image frame
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
            print('starting color is now {} {} at {}'.format(color_start,color_number_start, point_start))

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
                print('alpha1 is {} between {}{} and {}{}'.format(alpha1,color_start,color_number_start,color_second,color_number_second))
                #comupute Alpha 2 for the remaining color-
                for m in range(len(third_point_candidate)):
                    color_number_third = third_point_candidate[m][3]
                    alpha2_cand =  math.degrees(math.atan2(second_point_candidate[k][1]-third_point_candidate[m][1],second_point_candidate[k][0]- third_point_candidate[m][0]))
                    print('alpha2 is {} between {}{} and {}{}'.format(alpha2_cand, color_second,color_number_second,color_third,color_number_third))
                    # print('alpha2 is {}'.format(alpha2_cand))
                    if alpha2_cand > alpha1-self.AngleThreshold and alpha2_cand < alpha1 + self.AngleThreshold:
                        code_color = [color_start,color_second,color_third]
                        code_number = [color_number_start,color_number_second,color_number_third]
                        print(code_color)
                        print(code_number)
                        self.LineColorCombination.append(code_color)
                        self.LineNumberCombination.append(code_number)
                        self.drawCanvaLine(code_color,code_number)
                        


    def drawCanvaLine(self, color, number):
        self.TablecurrentCanvaLine = []
        table_index = []
        line_start_point = color[0]
        line_start_point_number = number[0]
        line_stop_point = color[2]
        line_stop_point_number = number[2]

        #find the starting point coordinates in the table
        for i in range(len(self.PointTableWithColor)):
            if line_start_point in self.PointTableWithColor[i]:
               if line_start_point_number in self.PointTableWithColor[i]:
                x1 = self.PointTableWithColor[i][0]
                y1 = self.PointTableWithColor[i][1]
        #find the ending point coordinates in the table
        for i in range(len(self.PointTableWithColor)):
            if line_stop_point in self.PointTableWithColor[i]:
               if line_stop_point_number in self.PointTableWithColor[i]:
                x2= self.PointTableWithColor[i][0]
                y2 = self.PointTableWithColor[i][1]

        self.TablecurrentCanvaLine.append(self.app.canvas.create_line(x1, y1, x2, y2, fill='red'))
        # canvas.create_line(15, 25, 200, 25)
    #

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

        print('dictionnary created with {} elements'.format(self.TotNumbOfImage))



class Child_window:
    def __init__(self, master, ImageId=0):
        self.master = master
        self.x = 0
        self.y =0
        self.clk = False
        # master.minsize(width=1280, height=1024)
        self.frame = tk.Frame(self.master)
        self.master.title("Frame number {}".format(ImageId))
        imageinfo = Image.open("framesLeft/frameL{}.jpg".format(ImageId))
        self.img = ImageTk.PhotoImage(Image.open("framesLeft/frameL{}.jpg".format(ImageId)))
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