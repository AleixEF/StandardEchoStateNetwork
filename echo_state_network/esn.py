#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STANDARD ECHO STATE NETWORK
A class with two main methods: teacher_forcing and
predict. The user can create an echo_state_network object with the class and 
use the methods to learn a signal and predict its continuation. 


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
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error


class EchoStateNetwork(object):
    def __init__(self, leaking_rate=0.9, num_neur=1000, conn_per_neur=10, 
                 spectr_rad=0.8, u_input=0.1, seed=None):
        
        if seed:                                                             
            np.random.seed(seed)
    
        self.leaking_rate = leaking_rate
        self.num_neur = num_neur
        self.u_in = u_input # a constant scalar value
        self.w_in = np.random.uniform(-1, 1, size=num_neur)
        self.w_res = self.build_reservoir(conn_per_neur, spectr_rad)
        self.w_fb = np.random.uniform(-1, 1, size=num_neur) 
        self.w_out = np.zeros(num_neur) #the vector we will use to predict
        self.xstate = np.zeros(num_neur)
        
    def teacher_forcing(self, train_signal, num_skip=1, beta_regul=0,
                                                               penalties=None):
        num_teaching = train_signal.shape[0]
        xstates_evol = np.zeros((num_teaching, 1+self.num_neur))
        #each row will be the concatenation of the u_input and the xstate
        #the idea is to build the sklearn arrays X and Y for ridge regression 
        xstates_evol[:, 0] = self.u_in
        for state_number, y in enumerate(train_signal):
            xstates_evol[state_number, 1:] = self.xstate
            self.update_state(y)
        
        #we exclude from the ridge regr the initial reservoir states that 
        # suffer a transcient period
        regr_xstates = xstates_evol[num_skip:, :]
        regr_ytargets = train_signal[num_skip:]
        if np.any(penalties):
            regr_penalties = penalties[num_skip:]
        else:
            regr_penalties = None
        
        self.w_out, train_mse = solve_linear_regression(regr_xstates,
                                     regr_ytargets, beta_regul, regr_penalties)                                                                                   
        return xstates_evol, train_mse
    
    def predict(self, pred_length):
        prediction = np.zeros(pred_length)
        for i in range(pred_length): #I have chosen a linear output
            reservoir_output = self.w_out[0]*self.u_in + \
                                            np.dot(self.w_out[1:], self.xstate) 
            prediction[i] = reservoir_output
            self.update_state(reservoir_output)
        return prediction
    
    def build_reservoir(self, conn_per_neur, spec_rad):
        w_res = np.zeros((self.num_neur, self.num_neur))
        for i in range(self.num_neur):
            random_columns = np.random.randint(0, self.num_neur, conn_per_neur)
            for j in random_columns:
                w_res[i, j] = np.random.uniform(-1, 1)
        w_res = change_spectral_radius(w_res, spec_rad)
        return w_res
    
    def update_state(self, output_neur):
        x_hat = np.dot(self.w_res, self.xstate) + self.w_fb*output_neur + \
        self.w_in*self.u_in
        x_hat = np.tanh(x_hat)
        self.xstate += self.leaking_rate*(x_hat-self.xstate)
        return
    

def change_spectral_radius(matrix, new_radius):
    eigenvalues = np.linalg.eig(matrix)[0]
    max_absolute_eigen = np.max(np.absolute(eigenvalues))
    return matrix * new_radius/max_absolute_eigen
        
def solve_linear_regression(x_states, y_targets, beta_regul, penalties):
    if beta_regul == 0:
        #fit intercept = False because the parameter u_in will play the role
        #of the intercept (the first column of xstates has the ctant val u_in)
        linear_reg = LinearRegression(fit_intercept=False)
        linear_reg.fit(x_states, y_targets, sample_weight=penalties)
        w_out = linear_reg.coef_
    else:
        ridge_reg = Ridge(alpha=beta_regul, fit_intercept=False)
        ridge_reg.fit(x_states, y_targets, sample_weight=penalties)
        w_out = ridge_reg.coef_
    y_pred = np.dot(x_states, w_out)
    train_mse = mean_squared_error(y_targets, y_pred)
    return w_out, train_mse
