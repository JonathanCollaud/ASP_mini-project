##Pitch corrector Voice Effect

This pitch corrector algorithm is organised in three python files:

* `main.py` : the main file that implement the pipelining of the different methods
* `methods.py` : contains the methods that are called from `main.py`
* `settings.py` : the settings that you can play with to adjust the voice effect

##Use

To launch the program, simply use the following command:

`python main.py

##### Advice for the settings:
If using mic as input_type, put headphones, there is no larsen protection
The parameters to change for FFT_size is the power of 2. Change the value x in "FFT_SIZE = 2**x * window_size"

Precisions in the comments of `settings.py`

##Test Folder
Folder with wav file to play with and some results of our algorithm. Another README inside.

##Figures
Figures from the report

##Requirements

* Python 3
* Matplotlib 3.1.1 (visualisation)
* Numpy 1.17.4 (math functions and array management)
* PyAudio 0.2.11 (audio stream management)
* Scipy 1.2 (interpolation and peak detection methods)
* Wave 0.0.2 (.wav file management)
* Wavio 0.0.4 (read .wav files)


## Authors
Damien Ronssin, damien.ronssin@epfl.ch 
Jonathan Collaud, jonathan.collaud@epfl.ch
