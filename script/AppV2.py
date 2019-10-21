from tkinter import *
from PIL import ImageTk, Image
import os

import tkinter as tk

Global_pos_x = 0
Global_pos_y = 0


class CommandWindow:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.button1 = tk.Button(self.frame, text = 'Open First Image', width = 25, command = self.new_window)
        self.button2 = tk.Button(self.frame, text = 'close', width = 25, command = self.close_image)
        self.button3 = tk.Button(self.frame, text = 'Next', width = 25, command = self.next_frame)
        self.button4 = tk.Button(self.frame, text = 'Previous', width = 25, command = self.prev_frame)
        self.button1.grid(row=0, column=0)
        self.button2.grid(row=1, column=0)
        self.button3.grid(row=2, column=0)
        self.button4.grid(row=3, column=0)
        self.frame.pack()
        self.currentImage = 0 #contain the frame number to pick in the set
        self.span = 10 # jump between frames to see the tool moving
        self.number_frame = 0 # diplayed frames count
        self.current_cursor_pos_X =0
        self.current_cursor_pos_Y  =0
        self.label_1 = Label(self.frame, text='Position red 1')
        string_to_display = Global_pos_x
        var1 = StringVar()
        var1.set(string_to_display)
        self.entry_1 = Entry(self.frame, textvariable = var1)

        self.label_1.grid(row=5, column=0)
        self.entry_1.grid(row=5, column=1)

        # self.imWindow = tk.Toplevel(self.master)
        # self.img = ImageTk.PhotoImage(Image.open("framesLeft/frameL{}.jpg".format(0)))

    def new_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo2(self.newWindow, ImageId=self.currentImage)

    def close_image(self):
        self.app.close_windows()

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

class Demo2:
    def __init__(self, master, ImageId=0):
        self.master = master
        # master.minsize(width=1280, height=1024)
        self.frame = tk.Frame(self.master)
        self.master.title("Frame number {}".format(ImageId))
        imageinfo = Image.open("framesLeft/frameL{}.jpg".format(ImageId))
        self.img = ImageTk.PhotoImage(Image.open("framesLeft/frameL{}.jpg".format(ImageId)))
        self.canvas = Canvas(self.master, width = imageinfo.size[0], height = imageinfo.size[1])
        print( imageinfo.size[0], imageinfo.size[1] )
        self.canvas.create_image(0,0, image=self.img, anchor="nw")
        self.canvas.pack()
        master.bind('<Button-1>', leftClick)


    def close_windows(self):
        self.master.destroy()

    # def next_frame(self):
    #     self.master.destroy()
    #     imageinfo = Image.open("framesLeft/frameL{}.jpg".format(10))
    #     self.img = ImageTk.PhotoImage(Image.open("framesLeft/frameL{}.jpg".format(10)))
    #     self.canvas = Canvas(self.master, width = imageinfo.size[0], height = imageinfo.size[1])
    #     self.canvas.create_image(0,0, image=self.img, anchor="nw")
    #     self.canvas.pack()

def leftClick(event):
    print("left")
    print('pos x {}'.format(event.x))
    print('pos y {}'.format(event.y))
    Global_pos_x = event.x
    Global_pos_y = event.y
    # return event.x, event.y


def main():
    root = tk.Tk()
    app = CommandWindow(root)
    root.mainloop()

if __name__ == '__main__':
    main()
    
# https://stackoverflow.com/questions/16115378/tkinter-example-code-for-multiple-windows-why-wont-buttons-load-correctly