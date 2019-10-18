from tkinter import *
from PIL import ImageTk, Image
import os

import tkinter as tk

class CommandWindow:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.button1 = tk.Button(self.frame, text = 'Open First Image', width = 25, command = self.new_window)
        self.button2 = tk.Button(self.frame, text = 'close', width = 25, command = self.close_image)
        self.button3 = tk.Button(self.frame, text = 'Next', width = 25, command = self.next_frame)
        self.button4 = tk.Button(self.frame, text = 'Previous', width = 25, command = self.prev_frame)
        self.button1.pack()
        self.button2.pack()
        self.button3.pack()
        self.button4.pack()
        self.frame.pack()
        self.currentImage = 10000
        self.span = 10
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
        imageinfo = Image.open("framesLeft/frameL{}.jpg".format(ImageId))
        self.img = ImageTk.PhotoImage(Image.open("framesLeft/frameL{}.jpg".format(ImageId)))
        self.canvas = Canvas(self.master, width = imageinfo.size[0], height = imageinfo.size[1])
        self.canvas.create_image(0,0, image=self.img, anchor="nw")
        self.canvas.pack()


    def close_windows(self):
        self.master.destroy()

    def next_frame(self):
        self.master.destroy()
        imageinfo = Image.open("framesLeft/frameL{}.jpg".format(10))
        self.img = ImageTk.PhotoImage(Image.open("framesLeft/frameL{}.jpg".format(10)))
        self.canvas = Canvas(self.master, width = imageinfo.size[0], height = imageinfo.size[1])
        self.canvas.create_image(0,0, image=self.img, anchor="nw")
        self.canvas.pack()

def main():
    root = tk.Tk()
    app = CommandWindow(root)
    root.mainloop()

if __name__ == '__main__':
    main()
    
# https://stackoverflow.com/questions/16115378/tkinter-example-code-for-multiple-windows-why-wont-buttons-load-correctly