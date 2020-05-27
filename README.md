# ClassicEchoStateNetwork
Contains a package with a standard reservoir computing algorithm that is used 
to predict the continuation of signals.

## System Requirements
Python 3.7 or newer, with the following libraries:
* Numpy
* Sklearn
* Matplotlib

## Instalation
The package echo_state_network contains a single module called esn.py, which has a class named EchoStateNetwork.
The main goal is to allow the user to import the EchoStateNetwork class in his Python programs. 
This can be done by just downloading the package and writing an import command at the beginning of any python 
file that needs the class:
```
from echo_state_network.esn import EchoStateNetwork
```
The user just needs to make sure that his Python files and the package are in the same directory, 
so that the package can always be found. 

## The EchoStateNetwork class
The class allows you to create an echo state network object,
which has a constructor and two main methods: **teacher forcing** and **predict**.
* The constructor builds the size of the esn from default values and initializes its random parameters. 
* The teacher forcing method receives the training signal and solves a ridge regression equation. 
* The predict method returns the prediction of the signal continuation.
