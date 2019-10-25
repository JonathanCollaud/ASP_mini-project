import pyaudio
import wave
import sys
import numpy as np

CHUNK = int(512)
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "voice.wav"


def processing(data):
    x = np.frombuffer(data, dtype=np.int16)
    x = np.fft.fft(x)
    z = shift_freq(x, 2)
    out = np.fft.ifft(z)
    out = np.real(out)
    return out

## test
def shift_freq(f_t_vect, corr_f):
    idx = np.arange(f_t_vect.shape[0])
    z = np.zeros_like(f_t_vect)
    reag = np.rint(idx/corr_f)
    reag = reag.astype(np.int16)
    z = z + f_t_vect[reag]
    return z




def play(stream, chunk):
    stream.write(chunk)

p = pyaudio.PyAudio()

in_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

out_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    chunk = in_stream.read(CHUNK)
    processed_chunk = processing(chunk)
    processed_chunk = processed_chunk.astype(np.int16).tostring()
    play(out_stream, processed_chunk)
    frames.append(processed_chunk)

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