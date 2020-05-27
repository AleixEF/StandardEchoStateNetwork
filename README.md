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
