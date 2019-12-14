import numpy as np
import pyaudio
import wave
import wavio

from methods import *
from settings import *

np.set_printoptions(4, suppress=True)

# Parallel and chunk size
PARALLEL_WINDOWS = int(1 / (1 - WINDOW_OVERLAP))
CHUNK_SIZE = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))

FORMAT = pyaudio.paInt16
CHANNELS = 1

# Build notes vectors
notes, notes_str = build_notes_vector(KEY, 4)

# Get window arrays
w_a = window(WINDOW_SIZE, WINDOW_OVERLAP, analysis_window_type)
w_s = window(WINDOW_SIZE, WINDOW_OVERLAP, synthesis_window_type)

# Number of 0 that will be padded
pad_size = int(FFT_SIZE - WINDOW_SIZE)

# Frequencies of real fft
freq = np.fft.rfftfreq(FFT_SIZE, 1.0/RATE)

# Info recap
print('Rate:', RATE)
print('Window:', WINDOW_SIZE)
print('FFT:', FFT_SIZE)
print('Overlap:', WINDOW_OVERLAP*100, '%')
print('Chunk:', CHUNK_SIZE)
print('Frequency Resolution:', freq[2]-freq[1], 'Hz')
print('Key:', KEY)
if input_type=='mic':
    print('Recording time:', RECORD_SECONDS, 's')

# Init pyaudio
p = pyaudio.PyAudio()

# Create output stream if needed
if play_sound:
    out_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK_SIZE)


if input_type=='sin':
    # Specify frequency of pure sin signal
    j = np.arange(RATE * RECORD_SECONDS)
    signal = np.array(6000 * np.sin(2 * np.pi * f * j / RATE), dtype=np.int16)
    n_iter = int(signal.shape[0]/CHUNK_SIZE)


elif input_type=='wav':
    wav_obj = wavio.read(wavefile_name)
    signal = wav_obj.data[:, 0]
    n_iter = int(signal.shape[0]/CHUNK_SIZE)


elif input_type=='mic':
    in_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
    n_iter = int((RATE / CHUNK_SIZE) * RECORD_SECONDS)

else:
    raise ValueError('Wrong input type, should be mic, wav or sin')

# List for input and output files
frames = []
frames_no_modif = []

# 2D arrays
chunk_array = np.zeros((PARALLEL_WINDOWS, CHUNK_SIZE), dtype=np.int16)
summed_chunks = np.zeros((PARALLEL_WINDOWS, CHUNK_SIZE), dtype=np.int16)

# Array to store the spectra (to visualize)
spectra = np.empty((int((RATE / CHUNK_SIZE) * RECORD_SECONDS), freq.shape[0]))

# Init of phase coherency array
Z = 1.0 + 0.0j


# For loop that will read the input signal in chunk
print("* start", flush=True)

for i in range(0, n_iter + PARALLEL_WINDOWS-1):
    # If statement to avoid reading for the last iterations (corresponding to delay due to windowing
    if i < n_iter:
        if input_type is not 'mic':
            chunk = signal[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE].copy()
        else:
            chunk = in_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            chunk = np.frombuffer(chunk, dtype=np.int16)

        frames_no_modif.append(chunk)

    else:
        chunk = np.zeros(CHUNK_SIZE)

    chunk_array[-1] = chunk

    # Flatten copies the array
    window = chunk_array.flatten()

    # Analysis window
    window = np.asarray(w_a * window)

    # Processing
    window, Z = processing(window, freq, Z, WINDOW_SIZE, CHUNK_SIZE, RATE,
               pad_size, notes, notes_str, i, plot=plot_window)

    # Synthesis window
    window = np.asarray(w_s * window)

    # Sum the overlapping windows
    summed_chunks = summed_chunks + window.reshape((PARALLEL_WINDOWS, CHUNK_SIZE))

    # Put the finished chunk to stream (index 0)
    out_chunk = summed_chunks[0].astype(np.int16).tostring()

    # To avoid adding to output file zero from initialisation of chunk arrays
    if i >= PARALLEL_WINDOWS - 1:
        if play_sound:
            play(out_stream, out_chunk)

        frames.append(out_chunk)

    chunk_array[:-1] = chunk_array[1:]                    
    summed_chunks[:-1] = summed_chunks[1:]
    summed_chunks[-1] = np.zeros((CHUNK_SIZE,))

print("* done              ", flush=True)

# Stop started streams
if input_type=='mic':
    in_stream.stop_stream()
    in_stream.close()

if play_sound:
    out_stream.stop_stream()
    out_stream.close()

p.terminate()

# Write input and output to wav file
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

wf = wave.open(WAVE_OUTPUT_FILENAME_NO_MODIF, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames_no_modif))
wf.close()


# Plot figure of time frequency domain if specified by user
if plot_end:
    input_signal = np.frombuffer(b''.join(frames_no_modif), dtype=np.int16)
    output_signal = np.frombuffer(b''.join(frames), dtype=np.int16)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    freq_all = np.fft.rfftfreq(input_signal.shape[0], 1.0/RATE)
    ax1.plot(freq_all, np.abs(np.fft.rfft(input_signal)))
    ax2.plot(freq_all, np.abs(np.fft.rfft(output_signal)))
    ax2.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude')
    ax2.set_ylabel('Amplitude')

    ax1.set_title('Input frequency spectrum')
    ax2.set_title('Output frequency spectrum')

    plt.draw()

    # Plot figure of time domain
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    ax1.plot(input_signal)
    ax2.plot(output_signal)
    ax1.set_title('Input waveform')
    ax2.set_title('Output waveform')
    ax2.set_xlabel('Sample index')
    ax1.set_ylabel('Amplitude')
    ax2.set_ylabel('Amplitude')
    plt.draw()


plt.show()

