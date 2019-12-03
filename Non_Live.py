import numpy as np
import pyaudio
from methods import *
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

record = False

KEY = 'C'
WINDOW_SIZE = int(8192)
WINDOW_OVERLAP = 0.75
PARALLEL_WINDOWS = int(1 / (1 - WINDOW_OVERLAP))
CHUNK_SIZE = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))
w_a = window(WINDOW_SIZE, WINDOW_OVERLAP, 'rect')
w_s = window(WINDOW_SIZE, WINDOW_OVERLAP, 'hann')

# 0 : no padding, 1: half signal half zeros ...
FFT_SIZE = 2**0 * WINDOW_SIZE

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "voice_modif.wav"
WAVE_OUTPUT_FILENAME_NO_MODIF = "voice_no_modif.wav"

notes, notes_str = build_notes_vector(KEY, 4)


pad_size = int(FFT_SIZE - WINDOW_SIZE)
# Print info
print('Rate:', RATE)
print('Window:', WINDOW_SIZE)
print('FFT:', FFT_SIZE)
print('Overlap:', WINDOW_OVERLAP*100, '%')
print('Chunk:', CHUNK_SIZE)
print('Key:', KEY)
print('Pad_size', pad_size)
print('Recording time:', RECORD_SECONDS, 's')


output = np.zeros(int(RATE * RECORD_SECONDS), dtype=np.int16)

# Frequencies of real fft
freq = np.fft.rfftfreq(FFT_SIZE, 1.0/RATE)

p = pyaudio.PyAudio()

if record:
    input = np.zeros(RATE * RECORD_SECONDS, dtype=np.int16)
    in_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    for i in range(0, int((RATE / CHUNK_SIZE) * RECORD_SECONDS)):
        chunk = in_stream.read(CHUNK_SIZE)
        chunk = np.frombuffer(chunk, dtype=np.int16)
        input[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] = chunk

    in_stream.stop_stream()
    in_stream.close()

else:
    n = np.arange(0, RATE*RECORD_SECONDS)
    f = 233
    input = 20000 * np.sin(2 * np.pi * f * n / RATE)
    #input = n * 100
    input = 10000*np.ones(RATE*RECORD_SECONDS)
    input = input.astype(np.int16)


write('input.wav', RATE, input)

chunk_array = np.zeros((PARALLEL_WINDOWS, CHUNK_SIZE), dtype=np.int16)
summed_chunks = np.zeros((PARALLEL_WINDOWS, CHUNK_SIZE), dtype=np.int16)

# Array to store the spectra (to visualize)
spectra = np.empty((int((RATE / CHUNK_SIZE) * RECORD_SECONDS), freq.shape[0]))

output_safe = np.zeros_like(output)

for i in range(0, int((RATE / CHUNK_SIZE) * RECORD_SECONDS)):

    chunk = input[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE].copy()
    print(chunk.shape)
    chunk_array[-1] = chunk.copy()

    # Flatten copies the array
    curr_window = chunk_array.flatten()

    # Analysis window
    curr_window = np.asarray(w_a * curr_window)
    # Processing
    #curr_window, y = processing(curr_window, freq, notes, WINDOW_SIZE, pad_size, notes_str, i, plot=True)

    # Store spectrum
    #spectra[i, :] = np.abs(y)

    # Synthesis window
    curr_window = np.asarray(w_s * curr_window)

    ### TODO sum chunks
    summed_chunks = (summed_chunks + curr_window.reshape((PARALLEL_WINDOWS, CHUNK_SIZE))).copy()

    out_chunk = summed_chunks[0].copy().astype(np.int16)

    output[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] = out_chunk.copy()

    chunk_array[:-1] = chunk_array[1:].copy()
    summed_chunks[:-1] = summed_chunks[1:].copy()
    summed_chunks[-1] = np.zeros((CHUNK_SIZE,))

"""
n_vect = np.arange(size)
signal = 10000 * np.sin(2 * np.pi * 474 * n_vect/RATE)
signal = signal.astype(np.int16)

#signal_pad =np.pad(signal, (0, pad_size), 'constant', constant_values=(0, 0))

y = np.fft.rfft(signal)
freq = np.fft.rfftfreq(size, 1.0/RATE)

shift_f = shift_factor(y, freq, notes, notes_str)
y_new = shift_freq(y, freq, shift_f)

signal_modif = np.fft.irfft(y_new)
signal_modif = signal_modif.astype(np.int16)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

ax1.plot(freq[200:400], np.abs(y[200:400]))
ax2.plot(freq[200:400], np.abs(y_new[200:400]))
fig.show()
"""

input_spec = np.abs(np.fft.rfft(input))
output_spec = np.abs(np.fft.rfft(output))
freq_tot = np.fft.rfftfreq(input.shape[0], 1.0/RATE)

fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
n_plot = 1000
ax1.plot(freq_tot[100:n_plot], input_spec[100:n_plot])
ax2.plot(freq_tot[100:n_plot], output_spec[100:n_plot])
ax1.set_title('Plot tot')
plt.draw()



write('output.wav', RATE, output)

#
# wf = wave.open('test_no_modif.wav', 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(16)
# wf.setframerate(RATE)
# wf.writeframes(input)
# wf.close()
#
# wf = wave.open('test_modif.wav', 'wb')
# wf.setnchannels(CHANNELS)
# wf.setsampwidth(16)
# wf.setframerate(RATE)
# wf.writeframes(b''.join(output))
# wf.close()

#fig = plt.figure()
#plt.plot(input)
#plt.plot(output)
#plt.draw()
plt.show()


