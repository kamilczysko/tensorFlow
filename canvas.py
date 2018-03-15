from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import netTest as test
import ctypes

b1 = "up"
xold, yold = None, None

root = Tk()
drawing_area = Canvas(root)
image1 = Image.new("RGB", (280, 280), (0, 0, 0))
draw = ImageDraw.Draw(image1)


def main():

    drawing_area.configure(width=280, height =280)
    drawing_area.pack()
    drawing_area.bind("<Motion>", motion)
    drawing_area.bind("<ButtonPress-1>", b1down)
    drawing_area.bind("<ButtonRelease-1>", b1up)
    root.configure(width = 500, height =500)
    button = Button(root, text='Test', command=test_image)
    button.pack(side=RIGHT)
    button = Button(root, text='Clear', command=clear_canvas)
    button.pack(side=LEFT)

    root.mainloop()

def b1down(event):
    global b1
    b1 = "down"           # you only want to draw when the button is down
                          # because "Motion" events happen -all the time-

def b1up(event):
    global b1, xold, yold
    b1 = "up"
    xold = None           # reset the line when you let go of the button
    yold = None

def motion(event):
    if b1 == "down":
        global xold, yold
        if xold is not None and yold is not None:
            event.widget.create_line(xold,yold,event.x,event.y, width=10)
            draw.line([xold,yold,event.x,event.y], (255,255,255), width=10)

        xold = event.x
        yold = event.y

def test_image():
    image1.thumbnail(size=(28,28))
    arr = np.array(image1.convert('L'))
    # converted_image = np.reshape(arr, newshape=(784,1))/255
    img = np.reshape(arr, newshape=(28, 28))
    pic = Image.fromarray(img)
    pic.show()
    guess = test.guess(arr)
    ctypes.windll.user32.MessageBoxW(0, guess, "answer", 1)

def clear_canvas():
    global image1
    global draw
    drawing_area.delete('all')
    image1 = Image.new("RGB", (280, 280), (0, 0, 0))
    draw = ImageDraw.Draw(image1)

if __name__ == "__main__":
    main()