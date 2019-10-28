import tkinter as tk
import os
import pandas as pd
from tkinter.ttk import Combobox
from PIL import Image, ImageTk

path = '../COCO/positionDataset/training'
window = tk.Tk()
data = ("on", "hanging", "above", "below", "inside", "around", "side", "side-up", "side-down")
cb = Combobox(window, values=data)
Positions = pd.Series([])

def setFeatures():
    try:

        if os.path.isfile(path + '/ImageDetails.csv'):
            df = pd.read_csv(path + "/ImageDetails.csv", sep=';')

            for index, row in df.iterrows():
                if (pd.isnull(row[3])):
                    imageid = int(row[0])
                    imageName = f"{imageid:012d}.png"
                    image = path + "/" + imageName
                    img = Image.open(image)

                    render = ImageTk.PhotoImage(img)

                    img = tk.Label(image=render)
                    img.image = render
                    img.place(x=0, y=0)

                    label = tk.Label(window, text='Choose the correct label: ')
                    label.place(x=730, y=160)

                    cb.place(x=730, y=180)

                    button = tk.Button(window, text='Add', fg='blue', command=lambda: callback(df, index, row))
                    button.config(width=20)
                    button.place(x=730, y=230)

                    window.title('Hello Python')
                    window.geometry("1000x500+10+10")
                    window.mainloop()
                    print("a")

    except FileNotFoundError as e:
        print(image + ' Not found')

def callback(df, index, row):
    val = cb.get()
    print(df.head())
    Positions[index] = val
    df.at[0, 'Position'] = val
    df.set_value(index,'Position', val)
    df.to_csv(path + '/ImageDetails.csv', encoding='utf-8', index=True, sep=';')
    print(df.at[0,'Position'])
    print(df.head())
    window.quit()
