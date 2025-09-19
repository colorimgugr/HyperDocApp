import joblib
import numpy as np
import glob
import os

wl=np.arange(400,1705,5)

for path in glob.glob(r"C:\Users\Usuario\Documents\GitHub\Hypertool\identification\data/*.joblib"):
    print('*******')
    model=joblib.load(path)
    print(path.split('\\')[-1])
    print(len(model['train_wl']))
    # joblib.dump(model_temp,path)
    # print(path)
    # model=joblib.load(path)
    # print(len(model['train_wl']))