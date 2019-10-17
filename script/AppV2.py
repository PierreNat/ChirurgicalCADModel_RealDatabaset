from tkinter import *
from PIL import ImageTk, Image
import os

import tkinter as tk

class CommandWindow:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.button1 = tk.Button(self.frame, text = 'Open Frist Image', width = 25, command = self.new_window)
        # self.button2 = tk.Button(self.frame, text = 'Next', width = 25, command = self.new_window)
        self.button1.pack()
        self.frame.pack()
        # self.imWindow = tk.Toplevel(self.master)
        # self.img = ImageTk.PhotoImage(Image.open("framesLeft/frameL{}.jpg".format(0)))

    def new_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo2(self.newWindow)

    # def next_image(self):
    #     self.master = master

class Demo2:
    def __init__(self, master):
        self.master = master
        # master.minsize(width=1280, height=1024)
        self.frame = tk.Frame(self.master)
        imageinfo = Image.open("framesLeft/frameL{}.jpg".format(0))
        self.img = ImageTk.PhotoImage(Image.open("framesLeft/frameL{}.jpg".format(0)))
        self.canvas = Canvas(self.master, width = imageinfo.size[0], height = imageinfo.size[1])
        self.canvas.create_image(0,0, image=self.img, anchor="nw")
        self.canvas.pack()


    def close_windows(self):
        self.master.destroy()

def main():
    root = tk.Tk()
    app = CommandWindow(root)
    root.mainloop()

if __name__ == '__main__':
    main()
    
# https://stackoverflow.com/questions/16115378/tkinter-example-code-for-multiple-windows-why-wont-buttons-load-correctly