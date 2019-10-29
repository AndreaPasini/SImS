import tkinter as tk
import os
import pandas as pd
from tkinter.ttk import Combobox
from PIL import Image, ImageTk

path = '../COCO/positionDataset/training'
pathImageDetail = path + '/ImageDetails.csv'
window = tk.Tk()
data = ("on", "hanging", "above", "below", "inside", "around", "side", "side-up", "side-down")
cb = Combobox(window, values=data)
Positions = pd.Series([])

def getImageName(imageId, pathFile):
    imageid = int(imageId)
    imageName = f"{imageid:012d}.png"
    image = pathFile + "/" + imageName
    return image

def setFeatures():
    try:
        if os.path.isfile(pathImageDetail):
            df = pd.read_csv(pathImageDetail, sep=';')
            for index, row in df.iterrows():
                if (pd.isnull(row[3])):
                    image = getImageName(row[0], path)
                    img = Image.open(image)
                    render = ImageTk.PhotoImage(img)
                    img = tk.Label(image=render)
                    img.image = render
                    img.place(x=0, y=0)
                    label = tk.Label(window, text='Choose the correct label: ')
                    label.place(x=730, y=160)
                    cb.place(x=730, y=180)
                    button = tk.Button(window, text='Add', fg='blue', command=lambda: callback(df, index))
                    button.config(width=20)
                    button.place(x=730, y=230)
                    window.title('Choose Label')
                    window.geometry("1000x500+10+10")
                    window.mainloop()
    except FileNotFoundError as e:
        print(image + ' Not found')


#TODO svuotare template immagine e dropdown
def callback(df, index):
    val = cb.get()
    Positions[index] = val
    df.at[index, 'Position'] = val
    df.to_csv(pathImageDetail, encoding='utf-8', index=False, sep=';')
    window.quit()
