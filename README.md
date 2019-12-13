##Pitch corrector Voice Effect

This pitch scaling algorithm is organised in three python files:

* `main.py` : the main file that implement the piplining of the different methods
* `methods.py` : contains the methods that are called from `test.py`
* `settings.py` : the settings that you can play with to adjust the voice effect

##Use

To launch the programm, simply use the follwing command:

`python test.py`

##Requirements

* Python 3.6
* Matplotlib 3.1.1 (graphical visualisation)
* Numpy 1.17.4 (math functions and array gestion)
* PyAudio 0.2.11 (audio stream gestion)
* Scipy 1.2 (interpolation and peak detection methods)
* Wave 0.0.2 (.wav file gestion)
* Wavio 0.0.4 (read .wav files)

