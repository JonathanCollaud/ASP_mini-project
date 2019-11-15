import pyaudio
import wave
import numpy as np
from methods import *

np.set_printoptions(4, suppress=True)


WINDOW_SIZE = int(512)
WINDOW_OVERLAP = 0.5
PARALLEL_WINDOWS = int(1 / (1 - WINDOW_OVERLAP))
CHUNK_SIZE = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))

# 0 : no padding, 1: half signal half zeros ...
FFT_SIZE = 2**0 * WINDOW_SIZE

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "voice_modif.wav"
WAVE_OUTPUT_FILENAME_NO_MODIF = "voice_no_modif.wav"


# Size of padding at the end
pad_size = int(FFT_SIZE - WINDOW_SIZE)

print('Padding:', pad_size)
# number of octave
n_oct = 4

# Name of notes
notes_str = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

# If we accept only A minor (C major) notes
notes_accepted = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
# gives this table
n_accepted_notes = np.array([0, 2, 3, 5, 7, 8, 10])
n_extended = np.array([])

for i in range(n_oct+1):
    n_extended = np.concatenate((n_extended, 12.0 * i + n_accepted_notes), axis=0)

notes_name = []
for i in range(n_oct+1):
    notes_name = notes_name + [n + str(i+1) for n in notes_accepted]


# Notes for our table of notes
notes = np.asarray(55.0*2.0**(n_extended/12.0))

# Frequencies of real fft
freq = np.fft.rfftfreq(FFT_SIZE, 1.0/RATE)

p = pyaudio.PyAudio()

in_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

out_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK_SIZE)

print("* recording")

frames = []
frames_no_modif = []

chunk_array = np.zeros((PARALLEL_WINDOWS, CHUNK_SIZE), dtype = np.int16)
summed_chunks = np.zeros((PARALLEL_WINDOWS, CHUNK_SIZE), dtype = np.int16)

for i in range(0, int((RATE / CHUNK_SIZE) * RECORD_SECONDS)):
    chunk = in_stream.read(CHUNK_SIZE)
    frames_no_modif.append(chunk)
    chunk = np.frombuffer(chunk, dtype=np.int16)

    chunk_array[-1] = chunk

    window = chunk_array.flatten() # flatten copies the array
    
    ### TODO analysis window
    
    window = processing(window, freq, notes, WINDOW_SIZE, pad_size, notes_name)

    ### TODO synthesis window

    ### TODO sum chunks
    summed_chunks = summed_chunks + window.reshape((PARALLEL_WINDOWS, CHUNK_SIZE))
        

    out_chunk = summed_chunks[0].astype(np.int16).tostring()
    play(out_stream, out_chunk)
    frames.append(out_chunk)
    chunk_array[:-1] = chunk_array[1:]                    
    summed_chunks[:-1] = summed_chunks[1:]
    summed_chunks[-1] = np.zeros((CHUNK_SIZE,), dtype = np.int16)

print("* done recording")

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
