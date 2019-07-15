import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import os

from attackgraph import file_op as fp
from attackgraph import json_op as jp

plt.figure()
# plt.title("Defender\'s Learning Curves")
plt.title("Attacker\'s Learning Curves")

# GASB_path = os.getcwd() + '/drawing/MGASB/def_data'
# RS_path = os.getcwd() + '/drawing/rew_shaping/def_data'
# Shapley_path = os.getcwd() + '/drawing/shapley/def_data'
# True_Shapley_path = os.getcwd() + '/drawing/true_shapley/def_data'

GASB_path = os.getcwd() + '/drawing/MGASB/att_data'
RS_path = os.getcwd() + '/drawing/rew_shaping/att_data'
Shapley_path = os.getcwd() + '/drawing/shapley/att_data'

GASB_data = []
RS_data = []
Shapley_data = []
# true_Shapley_data = []

for i in np.arange(2,12):
    G_data = fp.load_pkl(GASB_path+str(i)+'.pkl')
    R_data = fp.load_pkl(RS_path+str(i)+'.pkl')
    S_data = fp.load_pkl(Shapley_path+str(i)+'.pkl')
    # TS_data = fp.load_pkl(True_Shapley_path + str(i) + '.pkl')
    GASB_data.append(G_data)
    RS_data.append(R_data)
    Shapley_data.append(S_data)
    # true_Shapley_data.append(TS_data)

x_size = len(G_data)

GASB_data = np.array(GASB_data)
RS_data = np.array(RS_data)
Shapley_data = np.array(Shapley_data)
# true_Shapley_data = np.array(true_Shapley_data)


GASB_mean = np.mean(GASB_data, axis=0)
RS_mean = np.mean(RS_data, axis = 0)
S_mean = np.mean(Shapley_data, axis = 0)
# TS_mean = np.mean(true_Shapley_data, axis = 0)

# print(RS_mean[2000:2020])
# print(Arunesh_mean[2000:2020])

GASB_max = np.max(GASB_data, axis=0)
RS_max = np.max(RS_data, axis=0)
S_max = np.max(Shapley_data, axis=0)
# TS_max = np.max(true_Shapley_data, axis=0)

GASB_min = np.min(GASB_data, axis=0)
RS_min = np.min(RS_data, axis=0)
S_min = np.min(Shapley_data, axis=0)
# TS_min = np.min(true_Shapley_data, axis=0)

plt.grid()

X = np.linspace(1, x_size, x_size, endpoint=True)

plt.fill_between(X, GASB_max, GASB_min, alpha=0.1, color="r")
plt.fill_between(X, RS_max, RS_min, alpha=0.1, color="g")
plt.fill_between(X, S_max, S_min, alpha=0.1, color="orange")
# plt.fill_between(X, TS_max, TS_min, alpha=0.1, color="b")

plt.plot(X, GASB_mean, color="r", label='MGASB')
plt.plot(X, RS_mean, color="g", label="M.C.")
# plt.plot(X, TS_mean, color="b", label="S.V.")
plt.plot(X, S_mean, color="orange", label="S.V.")


plt.xlabel('Training time steps')
plt.ylabel('Average reward over 250 episodes')

plt.legend(loc="best")

plt.show()