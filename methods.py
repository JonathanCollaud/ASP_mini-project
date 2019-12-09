import pyaudio
import wave
import numpy as np
from scipy import interpolate as interp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
THRESHOLD = 200


def spline(x, y, x_new):
    tck_r = interp.splrep(x, np.real(y), s=0)
    tck_i = interp.splrep(x, np.imag(y), s=0)
    y_new_r = interp.splev(x_new, tck_r, der=0)
    y_new_i = interp.splev(x_new, tck_i, der=0)
    y_new = y_new_r + 1j * y_new_i
    return y_new


def interp1d(x, y, x_new):
    f_r = interp.interp1d(x, np.real(y), bounds_error=False, fill_value=0.0)
    f_i = interp.interp1d(x, np.imag(y), bounds_error=False, fill_value=0.0)
    out = f_r(x_new) + 1j * f_i(x_new)
    return out


def shift_factor(y, freq, notes, notes_name):
    """ Detect the fundamental frequency and compute the shift factor.
    :param y: Fourier transform (get via np.rfft)
    :param freq: Frequency corresponding to each index (get via np.rfftfreq)
    :param notes: notes table containing all notes (defined at beginning of code
    :return: shift factor
    """
    range_min = np.argmin(np.abs(freq - notes[0]))
    range_max = np.argmin(np.abs(freq - notes[-1]))
    peaks_idx, peaks_prop = find_peaks(np.abs(y[range_min:]),
                                       distance=120 / (freq[1] - freq[0]),
                                     prominence=60000)
    if peaks_idx.shape[0]>0:
        idx_fund = peaks_idx[0]
    else:
        idx_fund = np.argmax(y[range_min:range_max]*np.conj(y[range_min:range_max]))

    # if measured freq is 0, be careful, do not divide by 0
    polyfit = np.polyfit(freq[range_min + idx_fund - 1: range_min + idx_fund + 2], np.abs(y)[range_min + idx_fund - 1 : range_min + idx_fund + 2], 2)

    pitch = - polyfit[1] / (2 * polyfit[0])

    closest_note_idx = np.argmin(np.abs(pitch-notes))
    closest_note = notes[closest_note_idx]

    print("Closest note: ", repr(notes_name[closest_note_idx]), end='\r')
    # Computation of shift factor: coeff to apply to frequency of input signal
    shift_f = closest_note / pitch

    return shift_f, peaks_idx, pitch


def processing(x, freq, Z, window_size, step, rate, pad_size, notes, notes_name, i, plot=False):
    if not silence(x, THRESHOLD):

        # Zero Padding:
        x = np.pad(x, (0, pad_size), 'constant', constant_values=(0, 0))

        # Real fft
        y = np.fft.rfft(x)

        # Compute shift factor and peaks positions for phase coherency
        shift_f, peaks_idx, pitch = shift_factor(y, freq, notes, notes_name)

        if peaks_idx.shape[0]==0:
            delta_omega = 2*np.pi * pitch * (shift_f - 1)

        else:
            peaks_freq = freq[peaks_idx]
            peaks_freq = peaks_freq[:, None]
            closest_peak_idx = np.argmin(np.abs(freq-peaks_freq), axis=0)
            closest_peak_freq = peaks_freq[closest_peak_idx]
            closest_peak_freq = closest_peak_freq[:, 0]
            delta_omega = 2 * np.pi * closest_peak_freq * (shift_f - 1)

        Z = Z * np.exp(1j * delta_omega * step / rate)
        y_phase = Z * y

        # Shift frequency spectrum
        y_new = shift_freq(y_phase, freq, shift_f)

        # Inverse FFT
        out = np.fft.irfft(y_new)

        # Remove zero padding
        out = out[:window_size]

        if plot:
            if i % 20 == 0:
                fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
                xmin, xmax, ymin, ymax = 20, 20000, 200, 600000
                ax1.set_xlim([xmin, xmax])
                #ax1.set_ylim([ymin, ymax])
                ax2.set_xlim([xmin, xmax])
                #ax2.set_ylim([ymin, ymax])
                ax1.plot(freq, np.abs(y))
                ax1.plot(freq[peaks_idx], np.abs(y[peaks_idx]), 'o')
                ax2.plot(freq, np.abs(y_new))
                ax1.set_title('plot window')
                plt.draw()

    else:
        out = x
        Z = 1.0+0.0j
        print('Silence', end='\r')

    return out, Z


def shift_freq(y, freq, shift_f):
    # Interpolation: suppose you have correct pitch (freq_x = shift_f * freq) and resample to freq scale
    # to then do inverse fourier transform
    y_new = interp1d(shift_f * freq, y, freq)

    # Interpolation outside the freq range put to 0
    if shift_f < 1.0:
        idx_out_range = freq/shift_f > freq[-1]
        y_new[idx_out_range] = 0
    return y_new


def play(stream, chunk):
    stream.write(chunk)


def silence(x, silence_threshold):
    amp_max = np.max(np.abs(x))
    return amp_max < silence_threshold


def window(w_size, overlap=0.5, type='sine'):
    if overlap==0.75:
        overlap_factor = 1.0/np.sqrt(2)
    elif overlap==0.5:
        overlap_factor = 1.0
    elif overlap==0.0:
        w = np.ones(w_size)
        return w
    else:
        raise ValueError('Not valid overlap, should be 0, 0.5 or 0.75')

    n = np.arange(w_size)

    if type=='sine':
        w = overlap_factor * np.sin((n+0.5)*np.pi/w_size)

    elif type=='hann':
        w = overlap_factor * np.sin((n+0.5)*np.pi/w_size)**2

    elif type=='rect':
        w = overlap_factor * np.ones(w_size)

    elif type=='hamming':
        w = np.hamming(w_size)

    else:
        raise ValueError('Not valid window type')

    return w


def build_notes_vector(key, n_oct=4):
    # Name of notes
    if 'b' in key:
        notes_str = ['A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab']
        start_idx = notes_str.index(key)
        notes_str = np.asarray(notes_str)
    else:
        notes_str = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        start_idx = notes_str.index(key)
        notes_str = np.asarray(notes_str)

    key_notes = np.sort(np.mod(start_idx + np.array([0, 2, 4, 5, 7, 9, 11]), 12))

    # key_notes_str = notes_str[np.mod(key_notes, 12)]

    n_extended = np.array([])

    for i in range(n_oct + 1):
        n_extended = np.concatenate((n_extended, 12 * i + key_notes), axis=0)

    all_notes_str_ex = []

    for i in range(n_oct + 1):
        #all_notes_str_ex = all_notes_str_ex + [n + str(i + 1) for n in notes_str]
        all_notes_str_ex = all_notes_str_ex + [n for n in notes_str]

    all_notes_str_ex = np.asarray(all_notes_str_ex)
    notes_str_ex = all_notes_str_ex[n_extended.astype(np.int8)]
    notes_str_ex = np.asarray(notes_str_ex)

    # Notes for our table of notes, starting at 55Hz (A1)
    notes = np.asarray(55.0 * 2.0 ** (n_extended / 12.0))
    return notes, notes_str_ex
