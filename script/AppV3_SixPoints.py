from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import json

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
        self.TotNumbOfImage = 10 # each x frame will be picked, ideally 1000 for the ground truth database
        self.createDict()

        self.current_cursor_pos_X = 0
        self.current_cursor_pos_Y  = 0


        self.color2edit = 'Red'
        self.axis2edit = 'x'
        self.var2edit = '1'
        self.LabelSave = Label(self.frame, text='Dictionnary saved')


        self.v = StringVar()
        self.print_Status()
        self.val_x = StringVar()
        self.val_x.set(0)
        self.val_y = StringVar()
        self.val_y.set(0)

        #Red dot 1 ----------------------------------------------------------------------------

        self.Label_Rx1 = Label(self.frame, text='Red_1 x')
        self.Entry_Rx1 = Entry(self.frame, width = 5, textvariable = self.val_x)
        self.Label_Rx1 .grid(row=5, column=0)
        self.Entry_Rx1 .grid(row=5, column=1)

        self.Label_Ry1 = Label(self.frame, text='Red_1 y')
        self.Entry_Ry1 = Entry(self.frame,width = 5, textvariable = self.val_y)
        self.Label_Ry1 .grid(row=5, column=3)
        self.Entry_Ry1 .grid(row=5, column=4)

        self.buttonEdit_R1 = tk.Button(self.frame, text='edit', width=10, command=self.clearPose('R','1'))
        self.buttonVal_R1 = tk.Button(self.frame, text='val', width=10, command=self.valPose)

        self.buttonEdit_R1.grid(row=5, column=5)
        self.buttonVal_R1.grid(row=5, column=6)

        #Red dot 2 ----------------------------------------------------------------------------

        self.Label_Rx2 = Label(self.frame, text='Red_2 x')
        self.Entry_Rx2 = Entry(self.frame, width = 5, textvariable = self.val_x)
        self.Label_Rx2 .grid(row=6, column=0)
        self.Entry_Rx2 .grid(row=6, column=1)

        self.Label_Ry1 = Label(self.frame, text='Red_2 y')
        self.Entry_Ry1 = Entry(self.frame,width = 5, textvariable = self.val_y)
        self.Label_Ry1 .grid(row=6, column=3)
        self.Entry_Ry1 .grid(row=6, column=4)

        self.buttonEdit_R2 = tk.Button(self.frame, text='edit', width=10, command=self.clearPose('R','2'))
        self.buttonVal_R2 = tk.Button(self.frame, text='val', width=10, command=self.valPose)

        self.buttonEdit_R2.grid(row=6, column=5)
        self.buttonVal_R2.grid(row=6, column=6)

        # Green dot 1 ----------------------------------------------------------------------------
        self.Label_Gx1 = Label(self.frame, text='Green_1 x')
        self.Entry_Gx1 = Entry(self.frame, width=5, textvariable=self.val_x)
        self.Label_Gx1.grid(row=7, column=0)
        self.Entry_Gx1.grid(row=7, column=1)

        self.Label_Gy1 = Label(self.frame, text='Green_1 y')
        self.Entry_Gy1 = Entry(self.frame, width=5, textvariable=self.val_y)
        self.Label_Gy1.grid(row=7, column=3)
        self.Entry_Gy1.grid(row=7, column=4)

        self.buttonEdit_G1 = tk.Button(self.frame, text='edit', width=10, command=self.clearPose('G','1'))
        self.buttonVal_G1 = tk.Button(self.frame, text='val', width=10, command=self.valPose)

        self.buttonEdit_G1.grid(row=7, column=5)
        self.buttonVal_G1.grid(row=7, column=6)

        # Green dot 2 ----------------------------------------------------------------------------
        self.Label_Gx2 = Label(self.frame, text='Green_2 x')
        self.Entry_Gx2 = Entry(self.frame, width=5, textvariable=self.val_x)
        self.Label_Gx2.grid(row=8, column=0)
        self.Entry_Gx2.grid(row=8, column=1)

        self.Label_Gy1 = Label(self.frame, text='Green_2 y')
        self.Entry_Gy1 = Entry(self.frame, width=5, textvariable=self.val_y)
        self.Label_Gy1.grid(row=8, column=3)
        self.Entry_Gy1.grid(row=8, column=4)

        self.buttonEdit_G2 = tk.Button(self.frame, text='edit', width=10, command=self.clearPose('G','2'))
        self.buttonVal_G2 = tk.Button(self.frame, text='val', width=10, command=self.valPose)

        self.buttonEdit_G2.grid(row=8, column=5)
        self.buttonVal_G2.grid(row=8, column=6)

        # Blue dot 1 ----------------------------------------------------------------------------
        self.Label_Bx1 = Label(self.frame, text='Blue_1 x')
        self.Entry_Bx1 = Entry(self.frame, width=5, textvariable=self.val_x)
        self.Label_Bx1.grid(row=9, column=0)
        self.Entry_Bx1.grid(row=9, column=1)

        self.Label_By1 = Label(self.frame, text='Blue_1 y')
        self.Entry_By1 = Entry(self.frame, width=5, textvariable=self.val_y)
        self.Label_By1.grid(row=9, column=3)
        self.Entry_By1.grid(row=9, column=4)

        self.buttonEdit_B1 = tk.Button(self.frame, text='edit', width=10, command=self.clearPose('B','1'))
        self.buttonVal_B1 = tk.Button(self.frame, text='val', width=10, command=self.valPose)

        self.buttonEdit_B1.grid(row=9, column=5)
        self.buttonVal_B1.grid(row=9, column=6)

        # Blue dot 2 ----------------------------------------------------------------------------
        self.Label_Bx2 = Label(self.frame, text='Blue_2 x')
        self.Entry_Bx2 = Entry(self.frame, width=5, textvariable=self.val_x)
        self.Label_Bx2.grid(row=10, column=0)
        self.Entry_Bx2.grid(row=10, column=1)

        self.Label_By1 = Label(self.frame, text='Blue_2 y')
        self.Entry_By1 = Entry(self.frame, width=5, textvariable=self.val_y)
        self.Label_By1.grid(row=10, column=3)
        self.Entry_By1.grid(row=10, column=4)

        self.buttonEdit_B2 = tk.Button(self.frame, text='edit', width=10, command=self.clearPose('B','2'))
        self.buttonVal_B2 = tk.Button(self.frame, text='val', width=10, command=self.valPose)

        self.buttonEdit_B2.grid(row=10, column=5)
        self.buttonVal_B2.grid(row=10, column=6)


        self.app_created = False #true if child is created
        self.updatePose = False # can the position be updated

        self.frame.pack()



    def print_Status(self):
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

        self.LabelSave.grid(row=6, column=0)


    def new_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Child_window(self.newWindow, ImageId=self.currentFrameId)
        self.app_created = True
        self.app.canvas.bind('<Motion>', self.motion_all)
        self.val_x.set(self.AllDataPoint[self.number_frame]['Redx1'])
        self.val_y.set(self.AllDataPoint[self.number_frame]['Redy1'])
        self.Entry_Rx1 = Entry(self.frame, textvariable=self.val_x)
        self.Entry_Ry1 = Entry(self.frame, textvariable=self.val_y)

    def close_image(self):
        self.app.close_windows()
        self.app_created = False
        # self.frame.bind('<Motion>', self.motion)

    # def get_coordinate(self):
    #     self.current_cursor_pos_X, self.current_cursor_pos_Y = self.app.
    #     print(self.current_cursor_pos_X, self.current_cursor_pos_Y)

    def next_frame(self):
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
            self.val_x.set(self.AllDataPoint[self.number_frame]['Redx1'])
            self.val_y.set(self.AllDataPoint[self.number_frame]['Redy1'])
            self.Entry_Rx1 = Entry(self.frame, textvariable=self.val_x)
            self.Entry_Ry1 = Entry(self.frame, textvariable=self.val_y)
            self.print_Status()

    def prev_frame(self):
        if self.app_created:
            self.updatePose = True
            if (self.currentFrameId-self.span >= 0):
                self.currentFrameId = self.currentFrameId-self.span
                self.number_frame = self.number_frame - 1
            else:
                self.currentFrameId = self.currentFrameId
            self.close_image()
            self.new_window()
            self.val_x.set(self.AllDataPoint[self.number_frame]['Redx1'])
            self.val_y.set(self.AllDataPoint[self.number_frame]['Redy1'])
            self.Entry_Rx1 = Entry(self.frame, textvariable=self.val_x)
            self.Entry_Ry1 = Entry(self.frame, textvariable=self.val_y)
            print(self.number_frame)
            self.print_Status()


    def motion_all(self,event):
        if self.app_created:
            if self.updatePose:
                if self.app.clk:

                    self.val_x.set(self.app.x)
                    self.Entry_Rx1 = Entry(self.frame, textvariable = self.val_x)
                    self.AllDataPoint[self.number_frame]['Redx1'] =self.app.x
                    print('point saved for x of frame {} is {}'.format(self.number_frame, self.AllDataPoint[self.number_frame]['Redx1']))
                    self.val_y.set(self.app.y)
                    self.Entry_Ry1 = Entry(self.frame, textvariable = self.val_y)
                    self.AllDataPoint[self.number_frame]['Redy1'] =self.app.y
                    self.AllDataPoint[self.number_frame]['FrameId'] = self.currentFrameId
                    print('point saved for y of frame {} is {}'.format(self.number_frame, self.AllDataPoint[self.number_frame]['Redy1']))
                    self.app.clk = False


    # def update_field(self,event,color, axis,number):
    #
    #
    #
    # def current_val_field(self,event,color, axis,number):
    #     # color: R,G,B, axis: x,z number:1,2
    #     self.currentfield = color
    #     self.axis = axis
    #     self.number = number


    def clearPose(self, currentColor,currentNumber):

        self.val_x.set(99)
        entry_X = "self.Entry_{}x{}".format(currentColor,currentNumber)
        entry_Y = "self.Entry_{}y{}".format(currentColor, currentNumber)
        test = Entry(self.frame, textvariable=self.val_x)
        exec(entry_X  + " = '{}'".format(test))
        # exec(entry_Y + " = '{}'".format(99))
        # self.updatePose = True


        # self.val_x.set(7)
        # self.Entry_Rx1 = Entry(self.frame, textvariable=self.val_x)
        # self.val_y.set(7)
        # self.Entry_Ry1 = Entry(self.frame, textvariable=self.val_y)
        # self.updatePose = True


    def valPose(self):
        self.updatePose = False




    def createDict(self):

        self.AllDataPoint = []

        for i in range(0, self.TotNumbOfImage):  # creation of the list of dictionnary
            OneFrameDict = {
                "ImageNo": i,
                "Span": self.span,
                "FrameId":0,
                "Redx1": 0,
                "Redy1": 0,
                "Redx2": 0,
                "Redy2": 0,
                "Greenx1": 0,
                "Greeny1": 0,
                "Greenx2": 0,
                "Greeny2": 0,
                "Bluex1": 0,
                "Bluey1": 0,
                "Bluex2": 0,
                "Bluey2": 0,
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