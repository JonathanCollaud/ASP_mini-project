import numpy as np

from methods import *

# Define input type: 'mic', 'wav' or 'sin'
input_type = 'wav'

# If input_type is wav, specify path to a wav file
wavefile_name = 'tests/queen_F.wav'

# If play sound live while recording/processing
# If using mic input: no feedback protection, use headphones
play_sound = True

# Display or not the plot of time domain and frequency domain of full input and output signals
plot_end = False

# Plot some windows input and output frequency spectrum,
# (every 100 iterations of main for loop, this can open a lot of matplotlib windows)
# SLOW DOWNS A LOT THE CODE, CANT DO LIVE WITH THIS SET TO TRUE
# Set play_sound to false to use it
plot_window = False

# Key: to shift towards notes that belong to the key
# If want to use all notes: KEY='chromatic'
KEY = 'F'

# Window size (usually a power of two between 1024 and 8192)
WINDOW_SIZE = int(4096)

# Overlap: 0 ,0.5 or 0.75
WINDOW_OVERLAP = 0.75

# To specfiy amount of zero padding: change power of 2 for FFT_SIZE
# Power of 2 = 0 :no padding, 1: half signal half zeros , 2: one quarter signal three quarters 0 ...
# Good FFT size: 4096 or 8192
FFT_SIZE = 2**1 * WINDOW_SIZE

# Window functions
# Best setup: sine-sine
# If overlap 0: rect-rect automatically set
analysis_window_type = 'sine'
synthesis_window_type = 'sine'

# Set audio rate in Hz
RATE = 44100

# Threshold for silence function
SILENCE_THRESHOLD = 200

# Recording time if input_type is mic or sin
RECORD_SECONDS = 5

# Sinus frequency if input_type is sin
f = 466

# Output names
if input_type == 'sin':
    WAVE_OUTPUT_FILENAME = "sin_" + str(f) + "_W" + str(WINDOW_SIZE) + "_FFT" + str(FFT_SIZE) + "_O" + \
                           str(WINDOW_OVERLAP) + '_w_ ' + analysis_window_type + '_' + synthesis_window_type + ".wav"
    WAVE_OUTPUT_FILENAME_NO_MODIF = "sin_" + str(f) + ".wav"
elif input_type=='wav':
    WAVE_OUTPUT_FILENAME = wavefile_name + "_W" + str(WINDOW_SIZE) + "_FFT" + str(FFT_SIZE) + "_O" + \
                           str(WINDOW_OVERLAP) + ".wav"
    WAVE_OUTPUT_FILENAME_NO_MODIF = 'voice_no_modif.wav'
elif input_type=='mic':
    WAVE_OUTPUT_FILENAME = "mic_W" + str(WINDOW_SIZE) + "_FFT" + str(FFT_SIZE) + "_O" + \
                           str(WINDOW_OVERLAP) + ".wav"
    WAVE_OUTPUT_FILENAME_NO_MODIF = 'mic_no_modif.wav'
