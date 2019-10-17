from tkinter import *
from PIL import ImageTk, Image
import os

def motion(event):
    x, y = event.x, event.y
    print('{}, {}'.format(x, y))
    # root.unbind('<Motion>', motion)

def leftClick(event):
    print("left")
    # root.bind('<Motion>', motion)
    print(root.winfo_pointerx() - root.winfo_rootx())
    print(root.winfo_pointery() - root.winfo_rooty())
    leftframe = Frame(root)
    leftframe.pack(side=RIGHT)
    # lb1 = Listbox(leftframe)
    # lb1.insert(0,'test1')
    # lb1.insert(1,'test2')
    # lb1.pack()

def rightClick(event):
    print("right")

def callback():
    print("click")


def main():
    Number_frame = 3 #each x frame will be picked, ideally 1000 for the ground truth database
    FrameSpan = 5 #each x frame will be picked, ideally every 1000 frames

    AllDataPoint = []

    for i in range(0,Number_frame): #creation of the list of dictionnary
        OneFrameDict = {
            "FrameNo": i,
            "Redx1": 0,
            "Redx2": 0,
            "Redy1": 0,
            "Redy2": 0,
            "Greenx1": 0,
            "Greenx2": 0,
            "Greeny1": 0,
            "Greeny2": 0,
            "Bluex1": 0,
            "Bluex2": 0,
            "Bluey1": 0,
            "Bluey2": 0,
        }
        AllDataPoint.append(OneFrameDict)

    print(len(AllDataPoint))
    for i in range(0, Number_frame):
        frameId = i*FrameSpan
        print('frame {}/{}'.format(i+1,Number_frame))
        print('frame id {}'.format(frameId))
        AllDataPoint[i]['FrameNo'] = frameId



        root = Tk()
        img = ImageTk.PhotoImage(Image.open("framesLeft/frameL{}.jpg".format(frameId)))
        panel = Label(root, image = img)
        panel.pack(side = "bottom", fill = "both", expand = "yes")
        panel.pack()

        root.bind('<Button-1>',leftClick)
        root.bind('<Button-3>',rightClick)
        # b = Button(root, text="OK", command=callback)
        # b.pack()

        lb1 = Listbox(root)
        lb1.insert(0,'test1')
        lb1.insert(1,'test2')
        lb1.pack()
        root.mainloop()

    print('test result {}'.format(AllDataPoint[0]['FrameNo']))
    print('test result {}'.format(AllDataPoint[1]['FrameNo']))
    print('test result {}'.format(AllDataPoint[2]['FrameNo']))

if __name__ == '__main__':
    main()

# https://stackoverflow.com/questions/16115378/tkinter-example-code-for-multiple-windows-why-wont-buttons-load-correctly