import pyaudio
import numpy as np
from methods import *

# Define input type: mic, wav or sin
input_type = 'mic'

# If input_type is wav, specify path to wav file
wavefile_name = 'attention_1_f#.wav'

# If play sound live while recording/processing
play_sound = True

# Display or not the plot of time domain and frequency domain of the whole signal
plot_end = False

# Key: to shift towards notes that belong to the key
KEY = 'F#'

# Window size and overlap
WINDOW_SIZE = int(1024)
WINDOW_OVERLAP = 0.75

# Power of 2 = 0 :no padding, 1: half signal half zeros , 2: one quarter signal three quarters 0 ...
FFT_SIZE = 2**2 * WINDOW_SIZE

# Window functions
analysis_window_type = 'sine'
synthesis_window_type = 'sine'

# Output names
WAVE_OUTPUT_FILENAME = "voice_modif.wav"
WAVE_OUTPUT_FILENAME_NO_MODIF = "voice_no_modif.wav"

PARALLEL_WINDOWS = int(1 / (1 - WINDOW_OVERLAP))
CHUNK_SIZE = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))

# Set basic information for audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Recording time if input_type is mic
RECORD_SECONDS = 10

# Sinus frequency if input_type is sin
f = 466
