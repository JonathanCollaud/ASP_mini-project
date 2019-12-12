import pyaudio
import numpy as np
from methods import *

# Define input type: mic, wav or sin
input_type = 'wav'

# If input_type is wav, specify path to wav file
wavefile_name = 'attention_1_f#.wav'

# If play sound live while recording/processing
play_sound = False

# Display or not the plot of time domain and frequency domain of the whole signal
plot_end = True

# Key: to shift towards notes that belong to the key
KEY = 'F#'

# Window size and overlap
WINDOW_SIZE = int(2048)
WINDOW_OVERLAP = 0.75

# Power of 2 = 0 :no padding, 1: half signal half zeros , 2: one quarter signal three quarters 0 ...
FFT_SIZE = 2**2 * WINDOW_SIZE

# Window functions
analysis_window_type = 'sine'
synthesis_window_type = 'sine'


PARALLEL_WINDOWS = int(1 / (1 - WINDOW_OVERLAP))
CHUNK_SIZE = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))

# Set basic information for audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Recording time if input_type is mic
RECORD_SECONDS = 3

# Sinus frequency if input_type is sin
f = 116

# Output names
if input_type == 'sin':
    WAVE_OUTPUT_FILENAME = "sin_" + str(f) + "_W" + str(WINDOW_SIZE) + "_FFT" + str(FFT_SIZE) + "_O" + \
                           str(WINDOW_OVERLAP) + ".wav"
    WAVE_OUTPUT_FILENAME_NO_MODIF = "sin_" + str(f) + ".wav"

elif input_type=='wav':
    WAVE_OUTPUT_FILENAME = wavefile_name + "_W" + str(WINDOW_SIZE) + "_FFT" + str(FFT_SIZE) + "_O" + \
                           str(WINDOW_OVERLAP) + ".wav"
    WAVE_OUTPUT_FILENAME_NO_MODIF = 'voice_no_modif.wav'
