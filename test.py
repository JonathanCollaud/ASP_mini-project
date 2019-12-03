import pyaudio
import wave
import numpy as np
from methods import *


np.set_printoptions(4, suppress=True)


KEY = 'C'
WINDOW_SIZE = int(512)
WINDOW_OVERLAP = 0.75
PARALLEL_WINDOWS = int(1 / (1 - WINDOW_OVERLAP))
CHUNK_SIZE = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))

# 0 : no padding, 1: half signal half zeros ...
FFT_SIZE = 2**4 * WINDOW_SIZE

PURE_SIGNAL = False

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "voice_modif.wav"
WAVE_OUTPUT_FILENAME_NO_MODIF = "voice_no_modif.wav"

notes, notes_str = build_notes_vector(KEY, 4)

w_a = window(WINDOW_SIZE, WINDOW_OVERLAP, 'sine')
w_s = window(WINDOW_SIZE, WINDOW_OVERLAP, 'sine')

pad_size = int(FFT_SIZE - WINDOW_SIZE)
# Print info
print('Rate:', RATE)
print('Window:', WINDOW_SIZE)
print('FFT:', FFT_SIZE)
print('Overlap:', WINDOW_OVERLAP*100, '%')
print('Chunk:', CHUNK_SIZE)
print('Key:', KEY)
print('Recording time:', RECORD_SECONDS, 's')



# Frequencies of real fft
freq = np.fft.rfftfreq(FFT_SIZE, 1.0/RATE)

p = pyaudio.PyAudio()

in_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
out_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK_SIZE)


frames = []
frames_no_modif = []

chunk_array = np.zeros((PARALLEL_WINDOWS, CHUNK_SIZE), dtype=np.int16)
summed_chunks = np.zeros((PARALLEL_WINDOWS, CHUNK_SIZE), dtype=np.int16)

# Array to store the spectra (to visualize)
spectra = np.empty((int((RATE / CHUNK_SIZE) * RECORD_SECONDS), freq.shape[0]))
j = np.arange(RATE * RECORD_SECONDS)
signal = np.array(10000 * np.sin(2 * np.pi * 466 * j / RATE), dtype=np.int16)

print("* recording", flush=True)
for i in range(0, int((RATE / CHUNK_SIZE) * RECORD_SECONDS)):
    if PURE_SIGNAL:
        chunk = signal[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE].copy()
    else:
        chunk = in_stream.read(CHUNK_SIZE, exception_on_overflow=False)
        chunk = np.frombuffer(chunk, dtype=np.int16)
    frames_no_modif.append(chunk)

    chunk_array[-1] = chunk
    # Flatten copies the array
    window = chunk_array.flatten()

    # Analysis window
    window = np.asarray(w_a * window, dtype=np.int16)
    # Processing
    window, y = processing(window, freq, notes, WINDOW_SIZE, pad_size, notes_str)

    # Store spectrum
    spectra[i, :] = np.abs(y)

    # Synthesis window
    window = np.asarray(w_s * window, dtype=np.int16)

    summed_chunks = summed_chunks + window.reshape((PARALLEL_WINDOWS, CHUNK_SIZE))
        

    out_chunk = summed_chunks[0].astype(np.int16).tostring()
    #play(out_stream, out_chunk)
    frames.append(out_chunk)
    chunk_array[:-1] = chunk_array[1:]                    
    summed_chunks[:-1] = summed_chunks[1:]
    summed_chunks[-1] = np.zeros((CHUNK_SIZE,), dtype=np.int16)

print("* done recording", flush=True)

plot_spectra(freq, spectra, WINDOW_SIZE, RATE, WINDOW_OVERLAP)

in_stream.stop_stream()
in_stream.close()
out_stream.stop_stream()
out_stream.close()
p.terminate()

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

#plot_spectra(freq, spectra, WINDOW_SIZE, RATE, WINDOW_OVERLAP)
