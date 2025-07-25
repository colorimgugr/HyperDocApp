from PIL import Image
import os

folder=r'C:\Users\Usuario\Documents\DOC_Yannick\HYPERDOC Database_TEST\Samples\GT'
filename='00189-VNIR-mock-up_GT.png'

filepath=os.path.join(folder,filename)

im=Image.open(filepath)
im.getdata()