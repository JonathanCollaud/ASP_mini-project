import pyaudio
import wave
import numpy as np
from methods import *
import wavio
np.set_printoptions(4, suppress=True)

# Define input type: mic, wav or sin
input_type = 'mic'
play_sound = True

KEY = 'Eb'
WINDOW_SIZE = int(1024)
WINDOW_OVERLAP = 0.75
PARALLEL_WINDOWS = int(1 / (1 - WINDOW_OVERLAP))
CHUNK_SIZE = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))

# 0 : no padding, 1: half signal half zeros ...
FFT_SIZE = 2**3 * WINDOW_SIZE


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "voice_modif.wav"
WAVE_OUTPUT_FILENAME_NO_MODIF = "voice_no_modif.wav"

notes, notes_str = build_notes_vector(KEY, 4)

w_a = window(WINDOW_SIZE, WINDOW_OVERLAP, 'sine')
w_s = window(WINDOW_SIZE, WINDOW_OVERLAP, 'sine')

pad_size = int(FFT_SIZE - WINDOW_SIZE)

# Frequencies of real fft
freq = np.fft.rfftfreq(FFT_SIZE, 1.0/RATE)

# Print info
print('Rate:', RATE)
print('Window:', WINDOW_SIZE)
print('FFT:', FFT_SIZE)
print('Overlap:', WINDOW_OVERLAP*100, '%')
print('Chunk:', CHUNK_SIZE)
print('Frequency Resolution:', freq[2]-freq[1], 'Hz')
print('Key:', KEY)
print('Recording time:', RECORD_SECONDS, 's')

p = pyaudio.PyAudio()

if play_sound:
    out_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK_SIZE)


if input_type=='sin':
    j = np.arange(RATE * RECORD_SECONDS)
    signal = np.array(10000 * np.sin(2 * np.pi * 466 * j / RATE), dtype=np.int16)
elif input_type=='wav':
    wavefile_name = 'Maria_Victor_Refrain.wav'
    wav_obj = wavio.read(wavefile_name)
    signal = wav_obj.data[:, 0]
elif input_type=='mic':
    in_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

# List for input and output files
frames = []
frames_no_modif = []

chunk_array = np.zeros((PARALLEL_WINDOWS, CHUNK_SIZE), dtype=np.int16)
summed_chunks = np.zeros((PARALLEL_WINDOWS, CHUNK_SIZE), dtype=np.int16)

# Array to store the spectra (to visualize)
spectra = np.empty((int((RATE / CHUNK_SIZE) * RECORD_SECONDS), freq.shape[0]))

# Init of phase coherency coefficient
Z = 1.0 + 0.0j


print("* start", flush=True)

for i in range(0, int((RATE / CHUNK_SIZE) * RECORD_SECONDS)):
    if input_type is not 'mic':
        chunk = signal[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE].copy()
    else:
        chunk = in_stream.read(CHUNK_SIZE, exception_on_overflow=False)
        chunk = np.frombuffer(chunk, dtype=np.int16)

    frames_no_modif.append(chunk)
    chunk_array[-1] = chunk

    # Flatten copies the array
    window = chunk_array.flatten()

    # Analysis window
    window = np.asarray(w_a * window)

    # Processing
    window, Z = processing(window, freq, Z, WINDOW_SIZE, CHUNK_SIZE, RATE,
               pad_size, notes, notes_str, i, plot=False)

    # Synthesis window
    window = np.asarray(w_s * window)

    # Sum the overlapping windows
    summed_chunks = summed_chunks + window.reshape((PARALLEL_WINDOWS, CHUNK_SIZE))

    # Put the finished chunk to stream
    out_chunk = summed_chunks[0].astype(np.int16).tostring()

    if play_sound:
        play(out_stream, out_chunk)
    frames.append(out_chunk)
    chunk_array[:-1] = chunk_array[1:]                    
    summed_chunks[:-1] = summed_chunks[1:]
    summed_chunks[-1] = np.zeros((CHUNK_SIZE,))

print("* done", flush=True)

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

input_signal = np.frombuffer(b''.join(frames_no_modif), dtype=np.int16)
output_signal = np.frombuffer(b''.join(frames_no_modif), dtype=np.int16)

# Plot figure of frequency domain
fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
freq_all = np.fft.rfftfreq(input_signal.shape[0], 1.0/RATE)
ax1.plot(freq_all, np.abs(np.fft.rfft(input_signal)))
ax2.plot(freq_all, np.abs(np.fft.rfft(output_signal)))
ax1.set_title('Frequency Domain')
plt.draw()


# Plot figure of time domain
fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
ax1.plot(input_signal)
ax2.plot(output_signal)
ax1.set_title('Time domain')

plt.show()

