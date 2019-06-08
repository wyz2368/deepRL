import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import os

from attackgraph import file_op as fp
from attackgraph import json_op as jp

plt.rcdefaults()
fig, ax = plt.subplots(figsize=(10,5))

GASB_att = np.array([318.1, 304.8, 357.2, 361.1, 361.8, 305.6, 306.0, 340.0, 366.6, 341.7])
GASBpass_att = np.array([352.6, 365.0, 371.2, 329.5, 360.1, 366.8, 298.9, 323.7, 357.8, 359.5])
MGASB_att = np.array([432.5, 410.3, 451.2, 455.4, 377.7, 448.0, 330.8, 331.1, 434.5, 428.0])
RS_att = np.array([420.6, 434.9, 413.9, 390.0, 400.3, 430.3, 444.1, 327.7, 382.6, 449.6])

mean1 = np.mean(GASB_att)
mean2 = np.mean(GASBpass_att)
mean3 = np.mean(MGASB_att)
mean4 = np.mean(RS_att)

min1 = np.min(GASB_att)
min2 = np.min(GASBpass_att)
min3 = np.min(MGASB_att)
min4 = np.min(RS_att)

max1 = np.max(GASB_att)
max2 = np.max(GASBpass_att)
max3 = np.max(MGASB_att)
max4 = np.max(RS_att)


GASB_def = []
GASBpass_def = []
MGASB_def = np.array([-122.6, -115.6, -124.7, -123.8, -121.6, -118.0, -117.4, -117.6, -117.7, -126.5])
RS_def = np.array([-110.7, -108.5, -110.6, -112.0, -111.7, -112.4, -112.6, -112.9, -112.7, -111.6])

mean5 = np.mean(MGASB_def)
mean6 = np.mean(RS_def)

max5 = np.max(MGASB_def)
max6 = np.max(RS_def)

min5 = np.min(MGASB_def)
min6 = np.min(RS_def)

methods_def = ('MGASB', "MGASB with R.S")
y_pos_def = np.arange(len(methods_def))
performance_def = [mean5,mean6]
error_def = np.array([[mean5-min5,mean6-min6],[max5-mean5,max6-mean6]])


# Example data
methods = ('GASB','GASB w.o P.', 'MGASB', "MGASB with R.S")
y_pos = np.arange(len(methods))
performance = [mean1,mean2,mean3,mean4]
error = [[mean1-min1,mean2-min2,mean3-min3,mean4-min4],[max1-mean1,max2-mean2,max3-mean3,max4-mean4]]

# print(performance)
# print(error)

# print(y_pos)

ax.barh(y_pos, performance, xerr=error, color='orange', align='center', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(methods)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xticks(np.arange(0,500,step=50))
ax.set_xlabel('Expected Payoff')
ax.set_title('Attacker\'s Performance')
for i in np.arange(0,500,step=50):
    plt.axvline(x = i, linewidth=1, color='grey', linestyle='--' )

# ax.barh(y_pos_def, performance_def, height=0.4, xerr=error_def, color='orange', align='center', ecolor='black')
# ax.set_yticks(y_pos_def)
# ax.set_yticklabels(methods_def)
# ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xticks(np.arange(-150,0,step=20))
# ax.set_xlabel('Expected Payoff')
# ax.set_title('Defender\'s Performance')
# for i in np.arange(-150,0,step=20):
#     plt.axvline(x = i, linewidth=1, color='grey', linestyle='--' )

# plt.text(x=)

plt.show()