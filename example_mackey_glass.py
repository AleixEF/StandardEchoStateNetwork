#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LEARNING THE MACKEY GLASS
Small example of how to use the Echo State Network class given in the package.
We feed the esn with a common time delay signal called the Mackey glass: 
(Leon Glass and Michael Mackey (2010) 
href="http://www.scholarpedia.org/article/Mackey-Glass_equation"
(See also the header of the mackey_glass.txt file)
After the training process, the echo state network predicts the continuation
of the signal. The program prints the training mean squared error in the 
terminal and displays two figures: one with a plot of the mackey glass data
and another one with the prediction made by the esn.  
 
 
Copyright (C) 2020  Aleix Espuña Fontcuberta

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt

from echo_state_network.esn import EchoStateNetwork

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
esn = EchoStateNetwork()

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


