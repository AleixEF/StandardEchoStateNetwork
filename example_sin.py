#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 12:33:35 2020

@author: Aleix Espuña Fontcuberta
"""

import numpy as np
import matplotlib.pyplot as plt
from echo_state_networks.standard import StdEsn


time = np.linspace(0, 200, 2000)
sinus = np.sin(time)

time_step = time[1] - time[0]
pred_length = 1000
future_time = [time[-1] + i*time_step for i in range(1, pred_length+1)]

plt.figure()
plt.title("Sinus Training signal")
plt.plot(time, sinus)

#Echo state network construction with default parameters
esn = StdEsn()
#Training
xstates, train_mse = esn.teacher_forcing(sinus)
print("training MSE = ", train_mse)

#Prediction
prediction = esn.predict(pred_length)

plt.figure()
plt.title("Prediction of the continuation")
plt.plot(future_time, prediction, color="green")

