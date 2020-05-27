#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:55:05 2020

@author: Aleix Espuña Fontcuberta
"""

import numpy as np
import matplotlib.pyplot as plt

from echo_state_networks.standard import StdEsn

time = np.loadtxt("ExampleSignals/mackey_glass.txt", usecols=0)
mackey_glass = np.loadtxt("ExampleSignals/mackey_glass.txt", usecols=1)

pred_length = 1000 #we will predict 1000 points in the future.
#the time step for this signal is 1, so we know the future time instants:
time_future = [time[-1] + i for i in range(1, pred_length+1)] 

plt.figure()
plt.xlabel("time")
plt.title("Mackey glass training signal")
plt.plot(time, mackey_glass)

#Creation of the ESN with default parameters
esn = StdEsn()

#Training
#we add some noise to the mackey glass training signal as regularization
noise = np.random.uniform(0, 1e-6, size=len(mackey_glass)) 
train_signal = mackey_glass + noise
xstates, mse = esn.teacher_forcing(train_signal) 

print("Training mean squared error = ", mse)
                                                                                                                                      
#Prediction
prediction = esn.predict(pred_length)

plt.figure()
plt.title("Prediction of the continuation")
plt.xlabel("time")
plt.plot(time_future, prediction, color="green")

plt.show()


