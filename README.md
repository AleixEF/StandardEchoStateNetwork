# ClassicEchoStateNetwork
Contains a package with a standard reservoir computing algorithm that is used 
to predict the continuation of signals.

## System Requirements
Python 3.7 or newer, with the following libraries:
* Numpy
* Sklearn
* Matplotlib

## Installation
The package echo_state_network contains a single module called esn.py, which has a class named EchoStateNetwork.
The main goal of the installation is to allow the user to import the EchoStateNetwork class in his Python programs. 
This can be done by just downloading the package and writing an import command at the beginning of any python 
file that needs the class:
```
from echo_state_network.esn import EchoStateNetwork
```
The user just needs to make sure that his Python files and the package are in the same directory, 
so that the package can always be found. 

## Running the example files
This repository contains two example files: example_sin.py and example_mackey_glass.py.
They can be run with the simple commands ```python example_sin.py``` or ```python example_mackey_glass.py```.
The program example_sin.py learns a sinusoidal signal and its continuation is predicted. The program 
example_mackey_glass.py learns a more complicated signal, called the mackey glass (see
Leon Glass and Michael Mackey 2010 
 href=http://www.scholarpedia.org/article/Mackey-Glass_equation). 

In both programs the training mean squared error is printed and two matplotlib figures are displayed. 
The first figure plots the training signal used and the second figure plots the prediction of the continuation. 
Taking a look at these two example files is recommended, as their code is small and simple, which provides
a good insight into how to use the class.

## How to use the EchoStateNetwork class
To understand the mathematical meaning of the different parameters that will be mentioned, please see the file equations_doc.pdf included in this repository.
The class has a constructor and two main methods: **teacher forcing** and **predict**.

* The constructor builds the size of the esn from default values and initializes its random parameters. 
  By default it selects a reservoir of 1000 neurons, with a leaking rate equal to 0.9, a value u_in=0.1 and a
  spectral radius of 0.8 for the sparse reservoir matrix with 10 connections per neuron. The user can also fix a seed, to       initialize the esn with the same random parameters every time.
* The teacher forcing method is used to learn the signal.  
  ARGS:    
  **Training signal**  
  **num_skip** (optional): How many initial training data points are not included in the ridge regression problem. Takes the       value of 1 by default.  
  **beta** (optional): The regularization parameter in the ridge regression equation. Beta=0 by default.  
  **penalties** (optional): Array of weigths to penalize certain examples in the ridge regression. Default is None.  
RETURNS  
  **xstates**: Array of shape (num_samples, 1+num_neurons). Matrix containing the generated reservoir states in each row. The first column is the constant value u_in (hence the 1+num_neurons dimension).   
  **train_mse**: The training mean squared error.

* The predict method returns the prediction of the signal continuation.  
ARGS  
**pred_length**: How many points in the future will we predict.  
RETURNS  
**prediction**: Array of shape (pred_length,)





