##Pitch corrector Voice Effect

This pitch scaling algorithm is organised in three python files:

* `main.py` : the main file that implement the pipelining of the different methods
* `methods.py` : contains the methods that are called from `main.py`
* `settings.py` : the settings that you can play with to adjust the voice effect

##Use

To launch the program, simply use the following command:

`python main.py`

##Requirements

* Python 3
* Matplotlib 3.1.1 (graphical visualisation)
* Numpy 1.17.4 (math functions and array management)
* PyAudio 0.2.11 (audio stream management)
* Scipy 1.2 (interpolation and peak detection methods)
* Wave 0.0.2 (.wav file management)
* Wavio 0.0.4 (read .wav files)

