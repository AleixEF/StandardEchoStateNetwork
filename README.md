# StandardEchoStateNetwork
Contains a package with a reservoir computing algorithm that is used 
to predict the continuation of signals.

## System Requirements
Python 3.7 or newer, with the following libraries:
* Numpy
* Sklearn
* Matplotlib

## Installation
The package echo_state_network contains a single module called esn.py, which has a class named EchoStateNetwork.
If the user wants to use the class, he can simply import it in his Python scripts. 
This can be done by just writing an import command at the beginning of any python 
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
a good insight into how to use the class. For more documentation about the class, please visit the Wiki of this repository. I have also included a documentation file "equations.pdf", that explains what is an Echo State Network.  
