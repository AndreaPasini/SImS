from datetime import datetime
import tkinter as tk
import os
import numpy as np
import pandas as pd
from tkinter.ttk import Combobox
from PIL import Image, ImageTk
from image_analysis.ImageProcessing import getImageName
from image_analysis.DatasetUtils import createFolderByClass

### CONFIGURATION ###
path = '../COCO/positionDataset/training'
pathImageDetail = path + '/ImageDetails.csv'
groundPathImage = path + "/" + "groundTruth"
path = '../COCO/positionDataset/training'
pathImageDetail = path + '/ImageDetails.csv'
data = ("on", "hanging", "above", "below", "inside", "around", "side", "side-up", "side-down")
Positions = pd.Series([])
#####################

############ ACTION ###########
action = 'LABELS_FROM_FOLDERS'
#action = 'LABELING_GUI = False
###############################


def update_labels_from_folder_division():
    print("Update Images")
    df = pd.read_csv(pathImageDetail, sep=';')
    dirList = [item for item in os.listdir(groundPathImage) if os.path.isdir(os.path.join(groundPathImage, item))]
    for elem in dirList:
        classPath = groundPathImage + "/" + elem
        for file in os.listdir(classPath):
            id = file.lstrip("0").rstrip(".png")
            row = df.query('image_id == ' + id)
            originalFolder = row['Position'].values[0]
            if originalFolder == elem:
                continue
            else:
                index = row.index.values
                df.at[index[0], 'Position'] = elem
                df.to_csv(pathImageDetail, encoding='utf-8', index=False, sep=';')
                print("moved image " + file + " from " + originalFolder + " to " + elem)
    print("Update Completed")


def labeling_gui():
    window = tk.Tk()
    cb = Combobox(window, values=data)
    try:
        if os.path.isfile(pathImageDetail):
            df = pd.read_csv(pathImageDetail, sep=';')
            df = df.replace(np.nan, '', regex=True)
            for index, row in df.iterrows():
                if (row[3] == ''):
                    image = getImageName(row[0], path)
                    img = Image.open(image)
                    render = ImageTk.PhotoImage(img)
                    img = tk.Label(image=render)
                    img.image = render
                    img.place(x=0, y=0)
                    label = tk.Label(window, text='Choose the correct label: ')
                    label.place(x=730, y=160)
                    cb.place(x=730, y=180)
                    button = tk.Button(window, text='Add', fg='blue',
                                       command=lambda: callback(df, index, img, window, cb))
                    button.config(width=20)
                    button.place(x=730, y=230)
                    window.title(image)
                    window.geometry("1000x500+10+10")
                    window.mainloop()
    except FileNotFoundError as e:
        print(image + ' Not found')
    except RuntimeError as e:
        print(e)
    except AttributeError as e:
        print(e)


def callback(df, index, img, window, cb):
    val = cb.get()
    Positions[index] = val
    df.at[index, 'Position'] = val
    df.to_csv(pathImageDetail, encoding='utf-8', index=False, sep=';')
    img.config(image='')
    cb.set('')
    window.quit()


if __name__ == "__main__":
    start_time = datetime.now()
    print(start_time.strftime("Start date: %Y-%m-%d %H:%M:%S"))

    if action == 'LABELS_FROM_FOLDERS':
        update_labels_from_folder_division()
    elif action == 'LABELING_GUI':
        labeling_gui()
        createFolderByClass("Position")

    end_time = datetime.now()
    print("Done.")
    print('Duration: ' + str(end_time - start_time))
