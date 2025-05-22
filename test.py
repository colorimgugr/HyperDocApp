import numpy as np
import matplotlib.pyplot as plt

x=range(13)
f0=400
y=[]
for xi in x:
    y.append(f0*2**(xi/12))

plt.plot(x,y)