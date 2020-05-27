#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

LEARNING THE MACKEY GLASS
Small example of how to use the Echo State Network class given in the package.
We feed the esn with a simple sinusoidal signal. 
 After the training process, the echo state network predicts the continuation
 of the sinus. The program displays two figures: one with a plot of 
 the sinus data used for training and another one with the prediction.  
 
 
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


time = np.linspace(0, 200, 2000)
sinus = np.sin(time)

time_step = time[1] - time[0]
pred_length = 1000
future_time = [time[-1] + i*time_step for i in range(1, pred_length+1)]

plt.figure()
plt.title("Sinus Training signal")
plt.plot(time, sinus)

#Echo state network construction with default parameters
esn = EchoStateNetwork()
#Training
xstates, train_mse = esn.teacher_forcing(sinus)
print("training MSE = ", train_mse)

#Prediction
prediction = esn.predict(pred_length)

plt.figure()
plt.title("Prediction of the continuation")
plt.plot(future_time, prediction, color="green")

plt.show()

