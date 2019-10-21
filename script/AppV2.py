from tkinter import *
from PIL import ImageTk, Image
import os

import tkinter as tk

class CommandWindow:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.interface_creation()

    def interface_creation(self):
        self.button1 = tk.Button(self.frame, text = 'Open First Image', width = 20, command = self.new_window)
        self.button2 = tk.Button(self.frame, text = 'close', width = 20, command = self.close_image)
        self.button3 = tk.Button(self.frame, text = 'Next', width = 20, command = self.next_frame)
        self.button4 = tk.Button(self.frame, text = 'Previous', width = 20, command = self.prev_frame)
        self.button5 = tk.Button(self.frame, text = 'clear', width = 10, command = self.clearPose)
        self.button6 = tk.Button(self.frame, text = 'save all', width = 10, command = self.savePose)
        self.frame.bind('<Motion>', self.motion)
        self.button1.grid(row=0, column=0)
        self.button2.grid(row=1, column=0)
        self.button3.grid(row=2, column=0)
        self.button4.grid(row=3, column=0)

        self.frame.pack()
        self.currentImage = 0 #contain the frame number to pick in the set
        self.span = 10 # jump between frames to see the tool moving
        self.number_frame = 0 # diplayed frames count
        self.current_cursor_pos_X = 0
        self.current_cursor_pos_Y  = 0
        self.LabelRed1_x = Label(self.frame, text='Red 1 x pos')
        self.LabelRed1_y = Label(self.frame, text='Red 1 y pos')
        self.string_to_display = 10
        self.Red1_x = StringVar()
        self.Red1_x .set(self.string_to_display)
        self.Red1_y = StringVar()
        self.Red1_y .set(self.string_to_display)
        self.entry_Red1_x = Entry(self.frame, width = 5, textvariable = self.Red1_x)
        self.entry_Red1_y = Entry(self.frame,width = 5, textvariable = self.Red1_y)
        self.LabelRed1_x .grid(row=5, column=0)
        self.entry_Red1_x .grid(row=5, column=1)
        self.LabelRed1_y .grid(row=5, column=3)
        self.entry_Red1_y .grid(row=5, column=4)
        self.button5.grid(row=5, column=5)
        self.button6.grid(row=6, column=5)
        self.app_created = False


    def new_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Child_window(self.newWindow, ImageId=self.currentImage)
        self.app_created = True

    def close_image(self):
        self.app.close_windows()
        self.app_created = False

    # def get_coordinate(self):
    #     self.current_cursor_pos_X, self.current_cursor_pos_Y = self.app.
    #     print(self.current_cursor_pos_X, self.current_cursor_pos_Y)

    def next_frame(self):
        if (self.currentImage+self.span < 18000):
            self.currentImage = self.currentImage+self.span
        else:
            self.currentImage = self.currentImage
        self.close_image()
        self.new_window()
        self.number_frame = self.number_frame+1

    def prev_frame(self):
        if (self.currentImage-self.span > 0):
            self.currentImage = self.currentImage-self.span
        else:
            self.currentImage = self.currentImage
        self.close_image()
        self.new_window()


    def motion(self,event):
        if self.app_created:
            if self.app.clk:
                print('{}, {}'.format(self.app.x, self.app.y))
                self.app.clk = False
                self.Red1_x .set(self.app.x)
                self.entry_Red1_x = Entry(self.frame, textvariable = self.Red1_x)
                self.Red1_y .set(self.app.y)
                self.entry_Red1_y = Entry(self.frame, textvariable = self.Red1_y)

    def clearPose(self):
        self.Red1_x.set(0)
        self.entry_Red1_x = Entry(self.frame, textvariable=self.Red1_x)
        self.Red1_y.set(0)
        self.entry_Red1_y = Entry(self.frame, textvariable=self.Red1_y)


    def savePose(self):
        #save  in dictionnary
        a =2

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
        print( imageinfo.size[0], imageinfo.size[1] )
        self.canvas.create_image(0,0, image=self.img, anchor="nw")
        self.canvas.pack()
        master.bind('<Button-1>', self.clicked)
        # master.bind('<Motion>', self.motion)
        # print(self.mm.pos_x)
        # print(self.mm.pos_y)
        # master.bind('<Button-1>', leftClick)

    def close_windows(self):
        self.master.destroy()


    def clicked(self, event):
        self.x, self.y = event.x, event.y
        self.clk = True
        # print(x,y)

    # def motion(self,event):
    #     if self.clicked:
    #         print('{}, {}'.format(self.x, self.y))
    #         self.clicked = False




    # def next_frame(self):
    #     self.master.destroy()
    #     imageinfo = Image.open("framesLeft/frameL{}.jpg".format(10))
    #     self.img = ImageTk.PhotoImage(Image.open("framesLeft/frameL{}.jpg".format(10)))
    #     self.canvas = Canvas(self.master, width = imageinfo.size[0], height = imageinfo.size[1])
    #     self.canvas.create_image(0,0, image=self.img, anchor="nw")
    #     self.canvas.pack()

# def leftClick(event):
#     print("left")
#     print('pos x {}'.format(event.x))
#     print('pos y {}'.format(event.y))
#     Global_pos_x = event.x
#     Global_pos_y = event.y
    # return event.x, event.y


class MousePose():
  def __init__(self):
    self.pos_x = 0
    self.pos_y = 0


  def select(self, event):
      print("left")
      pos_x  = event.x
      pos_y  = event.y


def main():
    root = tk.Tk()
    app = CommandWindow(root)
    # Get an instance of the MousePos object

    root.mainloop()

if __name__ == '__main__':
    main()
    
# https://stackoverflow.com/questions/16115378/tkinter-example-code-for-multiple-windows-why-wont-buttons-load-correctly