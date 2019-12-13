import numpy as np

from methods import *

# Define input type: 'mic', 'wav' or 'sin'
input_type = 'sin'

# If input_type is wav, specify path to wav file
wavefile_name = 'attention_1_false.wav'

# If play sound live while recording/processing
play_sound = False

# Display or not the plot of time domain and frequency domain of full input and output signals
plot_end = True

# Plot some windows input and output frequency spectrum,
# (every 100 iterations of main for loop, this can open a lot of matplotlib windows)
# SLOW DOWNS A LOT THE CODE, CANT DO LIVE WITH THIS SET TO TRUE
# Set play_sound to false to use it
plot_window = False

# Key: to shift towards notes that belong to the key
KEY = 'A'

# Window size and overlap

WINDOW_SIZE = int(2048)
WINDOW_OVERLAP = 0.75
PARALLEL_WINDOWS = int(1 / (1 - WINDOW_OVERLAP))
CHUNK_SIZE = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))

# Power of 2 = 0 :no padding, 1: half signal half zeros , 2: one quarter signal three quarters 0 ...
FFT_SIZE = 2**0 * WINDOW_SIZE

# Window functions
analysis_window_type = 'sine'
synthesis_window_type = 'sine'

# Set basic information for audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Threshold for silence function
SILENCE_THRESHOLD = 200

# Recording time if input_type is mic
RECORD_SECONDS = 2

# Sinus frequency if input_type is sin
f = 466

# Output names
if input_type == 'sin':
    WAVE_OUTPUT_FILENAME = "sin_" + str(f) + "_W" + str(WINDOW_SIZE) + "_FFT" + str(FFT_SIZE) + "_O" + \
                           str(WINDOW_OVERLAP) + ".wav"
    WAVE_OUTPUT_FILENAME_NO_MODIF = "sin_" + str(f) + ".wav"
elif input_type=='wav':
    WAVE_OUTPUT_FILENAME = wavefile_name + "_W" + str(WINDOW_SIZE) + "_FFT" + str(FFT_SIZE) + "_O" + \
                           str(WINDOW_OVERLAP) + ".wav"
    WAVE_OUTPUT_FILENAME_NO_MODIF = 'voice_no_modif.wav'
elif input_type=='mic':
    WAVE_OUTPUT_FILENAME = "mic_W" + str(WINDOW_SIZE) + "_FFT" + str(FFT_SIZE) + "_O" + \
                           str(WINDOW_OVERLAP) + ".wav"
    WAVE_OUTPUT_FILENAME_NO_MODIF = 'mic_no_modif.wav'
