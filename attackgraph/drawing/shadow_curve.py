import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import os

from attackgraph import file_op as fp
from attackgraph import json_op as jp

plt.figure()
plt.title("Defender\'s Learning Curves")

GASB_path = os.getcwd() + '/drawing/MGASB/def_data'
RS_path = os.getcwd() + '/drawing/rew_shaping/def_data'

GASB_data = []
RS_data = []

for i in np.arange(2,12):
    G_data = fp.load_pkl(GASB_path+str(i)+'.pkl')
    R_data = fp.load_pkl(RS_path+str(i)+'.pkl')
    GASB_data.append(G_data)
    RS_data.append(R_data)

x_size = len(G_data)

GASB_data = np.array(GASB_data)
RS_data = np.array(RS_data)

# print(np.shape(GASB_data))

GASB_mean = np.mean(GASB_data, axis=0)
RS_mean = np.mean(RS_data, axis = 0)

GASB_max = np.max(GASB_data, axis=0)
RS_max = np.max(RS_data, axis=0)

GASB_min = np.min(GASB_data, axis=0)
RS_min = np.min(RS_data, axis=0)

# print(GASB_mean, RS_mean)
# print(GASB_std, RS_std)

# print(GASB_mean[:50])

plt.grid()

X = np.linspace(1, x_size, x_size, endpoint=True)

plt.fill_between(X, GASB_max, GASB_min, alpha=0.1, color="r")
plt.fill_between(X, RS_max, RS_min, alpha=0.1, color="g")

plt.plot(X, GASB_mean, color="r", label='MGASB')
plt.plot(X, RS_mean, color="g", label="MGASB with R.S.")

plt.legend(loc="best")

plt.show()