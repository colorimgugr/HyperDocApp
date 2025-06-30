import pandas as pd
import os
import sys

folder = r'C:\Users\Usuario\Documents\DOC_Yannick\App_present_24_06\datas\Archivo chancilleria_for_Registering/'
file_name = 'reg_test.h5'
filepath = folder + file_name
os.path.exists(filepath)

if getattr(sys, 'frozen', False):  # pynstaller case
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

lookup_table_path = os.path.join(BASE_DIR,
                                      "Hypertool/data_vizualisation/Spatially registered minicubes equivalence.csv")

df = pd.read_csv(lookup_table_path)

curent='00189-VNIR-mock-up'
otro=df.loc[df['VNIR']==curent]['SWIR'][0]

