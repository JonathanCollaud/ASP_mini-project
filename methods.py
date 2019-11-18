import pyaudio
import wave
import numpy as np
from scipy import interpolate as interp

THRESHOLD = 50


def spline(x, y, x_new):
    tck_r = interp.splrep(x, np.real(y), s=0)
    tck_i = interp.splrep(x, np.imag(y), s=0)
    y_new_r = interp.splev(x_new, tck_r, der=0)
    y_new_i = interp.splev(x_new, tck_i, der=0)
    y_new = y_new_r + 1j * y_new_i
    return y_new


def spline_polar(x, y, x_new):
    amp, phase = C2P(y)

    tck_r = interp.splrep(x, amp, s=0)
    tck_i = interp.splrep(x, phase, s=0)

    y_new_amp = interp.splev(x_new, tck_r, der=0)
    y_new_phase= interp.splev(x_new, tck_i, der=0)

    y_new = P2C(y_new_amp, y_new_phase)
    return y_new


def interp1d(x, y, x_new):
    f_r = interp.interp1d(x, np.real(y), bounds_error=False, fill_value=0.0)
    f_i = interp.interp1d(x, np.imag(y), bounds_error=False, fill_value=0.0)
    out = f_r(x_new) + 1j * f_i(x_new)
    return out


def interp1d_polar(x, y, x_new):
    amp, phase = C2P(y)
    f_amp = interp.interp1d(x, amp, bounds_error=False, fill_value=0.0)
    f_phase = interp.interp1d(x, phase, bounds_error=False, fill_value=0.0)
    out = P2C(f_amp(x_new), f_phase(x_new))
    return out


def shift_factor(y, freq, notes, notes_name):
    """ Compute the shift factor to apply to frequencies.
    :param y: Fourier transform (get via np.rfft)
    :param freq: Frequency corresponding to each index (get via np.rfftfreq)
    :param notes: notes table containing all notes (defined at beginning of code
    :return: shift factor
    """

    idx_max = np.argmax(y*np.conj(y))

    # if measured freq is 0, be careful, do not divide by 0
    if idx_max != 0:
        pitch = freq[idx_max]
    else:
        pitch = freq[1]

    closest_note_idx = np.argmin(np.abs(pitch-notes))
    closest_note = notes[closest_note_idx]
    print("Closest note: ", notes_name[closest_note_idx], end='\r')

    # Computation of shift factor: coeff to apply to frequency of input signal
    shift_f = closest_note / pitch
    return shift_f


def processing(x, freq, notes, chunk_size, pad_size, notes_name):
    if not silence(x, THRESHOLD):

        # Zero Padding:
        x = np.pad(x, (0, pad_size), 'constant', constant_values=(0, 0))
        x = x.astype(np.int16)

        # Real fft
        y = np.fft.rfft(x)

        # Compute shift factor
        shift_f = shift_factor(y, freq, notes, notes_name)
        #print(shift_f)

        # shift_f = 4/5
        # Shift frequency spectrum
        y_new = shift_freq(y, freq, shift_f)

        # y_new = y
        # Inverse FFT and take real part
        out = np.fft.irfft(y_new)

        # Remove zero padding
        out = out[:chunk_size]

        out = np.real(out)

    else:
        out = x

    return out


def shift_freq(y, freq, shift_f):
    # Interpolation: suppose you have correct pitch (freq_x = shift_f * freq) and resample to freq scale
    # to then do inverse fourier transform
    y_new = interp1d(shift_f * freq, y, freq)
    # y_new = y
    return y_new


def play(stream, chunk):
    stream.write(chunk)


def P2C(r, angles):
    """
    Take a complex number in polar form and put it in cartesian
    :param r: radius
    :param angles: phase in radian
    :return: complex number in cartesian coordinates
    """
    return r * np.exp(1j*angles)


def C2P(x):
    """
    Take a complex np array and return a tuple of np array with norm and phase (polar coordinates)
    :param x: complex np array
    :return: 2D array with norm and phase
    """
    return np.abs(x), np.angle(x)


def silence(x, silence_threshold):
    amp_max = np.max(np.abs(x))
    return amp_max < silence_threshold


def window(w_size, overlap=0.5, type='sine'):

    if overlap==0.75:
        overlap_factor = 1.0/np.sqrt(2)
    elif overlap==0.5:
        overlap_factor = 1.0

    else:
        raise ValueError('Not valid overlap, should be 0.5 of 0.75')

    n = np.arange(w_size)

    if type=='sine':
        w = overlap_factor * np.sin((n+0.5)*np.pi/w_size)

    elif type=='hann':
        w = overlap_factor * np.sin((n+0.5)*np.pi/w_size)**2

    elif type=='rect':
        w = overlap_factor * np.ones(w_size)

    else:
        raise ValueError('Not valid window type')

    return w


def build_notes_vector(key, n_oct):
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
        all_notes_str_ex = all_notes_str_ex + [n + str(i + 1) for n in notes_str]

    all_notes_str_ex = np.asarray(all_notes_str_ex)
    notes_str_ex = all_notes_str_ex[n_extended.astype(np.int8)]
    notes_str_ex = np.asarray(notes_str_ex)

    # Notes for our table of notes, starting at 55Hz (A1)
    notes = np.asarray(55.0 * 2.0 ** (n_extended / 12.0))
    return notes, notes_str_ex
