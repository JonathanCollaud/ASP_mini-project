import pyaudio
import wave
import numpy as np
from scipy import interpolate as interp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Threshold for silence function
THRESHOLD = 200


def interp1d_p(x, y, x_new, kind='linear'):
    """
    Interpolation of window spectrum, interpolates the norm of each complex number and take the phase of the nearest bin
    :param x: Shift_factor * freq_axis
    :param y: Fourier transform of window (rfft)
    :param x_new: freq_axis
    :param kind: (string) either 'linear' for piecewise linear interpolation or 'cubic' for cubic spline to interpolate
                 norm of fourier coefficients
    :return: interpolation of spectrum on tuned frequency (shift_f * freq_axis) to real fft frequency bins (freq_axis)
    """
    # Convert y in amplitude and phase representation
    amp, phase = C2P(y)
    # Interpolates amplitude
    f_amp = interp.interp1d(x, amp, kind=kind, bounds_error=False, fill_value=0.0, assume_sorted=True)
    # Interpolates phase (nearest phase to avoid problems with the phase discontinuity between -pi and pi
    f_phase = interp.interp1d(x, phase, kind='nearest', bounds_error=False, fill_value=0.0, assume_sorted=True)
    out = f_amp(x_new) * np.exp(1.0j * f_phase(x_new))
    return out


def processing(x, freq, Z, window_size, step, rate, pad_size, notes, notes_name, i, plot=False):
    """
    Return the window signal with the frequency shifted, as well as the phase-coherency array Z
    :param x: window array of sound signal
    :param freq: freq array with corresponding frequency for each index of rfft
    :param Z: Phase-coherency array
    :param window_size: number of sample in a window
    :param step: Size of a chunk: window_size/(1-overlap)
    :param rate: sampling rate of the signal (eg: 44100)
    :param pad_size: size of padding for fourier transform
    :param notes: array containing frequency of each notes in key
    :param notes_name: array containing names of each notes in key
    :param i: iteration index we on (used to plot at certain iteration only)
    :param plot: (boolean): if need to plot frequency spectrum of window and modified window
    :return: out: modified signal,
             Z: phase coherency array
    """

    # Check if there is a sound to modified or just no sound/weak background noise
    if not silence(x, THRESHOLD):

        # Zero Padding:
        x = np.pad(x, (0, pad_size), 'constant', constant_values=(0, 0))

        # Real fft
        y = np.fft.rfft(x)

        # Compute shift factor, peaks positions for phase coherency and pitch (fundamental frequency in window)
        shift_f, peaks_idx, pitch = shift_factor(y, freq, notes, notes_name)

        # If no peak detected, all phase shifted according to detected pitch frequency
        if peaks_idx.shape[0]==0:
            delta_omega = 2*np.pi * pitch * (shift_f - 1)

        # Normal case: phase are adjusted with the frequency of the closest peak for each frequency bin of fft
        else:
            peaks_freq = freq[peaks_idx]
            peaks_freq[0] = pitch
            peaks_freq = peaks_freq[:, None]
            closest_peak_idx = np.argmin(np.abs(freq-peaks_freq), axis=0)
            closest_peak_freq = peaks_freq[closest_peak_idx]
            closest_peak_freq = closest_peak_freq[:, 0]
            delta_omega = 2 * np.pi * closest_peak_freq * (shift_f - 1)

        # Update phasor Z
        Z = Z * np.exp(1j * delta_omega * step / rate)

        # Apply phase correction
        y_phase = Z * y

        # Shift frequency spectrum
        y_new = shift_freq(y_phase, freq, shift_f)

        # Inverse FFT
        out = np.fft.irfft(y_new)

        # Remove zero padding
        out = out[:window_size]

        # To plot frequency domain of orignal and modified window signal
        # (SLOWS DOWN THE CODE A LOT, LIVE IMPOSSIBLE)
        if plot:
            if i % 100 == 0:
                fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
                xmin, xmax, ymin, ymax = 20, 5000, 200, 600000
                ax1.set_xlim([xmin, xmax])
                ax2.set_xlim([xmin, xmax])
                ax1.plot(freq, np.abs(y))
                ax1.plot(freq[peaks_idx], np.abs(y[peaks_idx]), 'o')
                ax2.plot(freq, np.abs(y_new))
                ax2.set_xlabel('Frequency (Hz)')
                ax1.set_ylabel('Amplitude')
                ax2.set_ylabel('Amplitude')

                ax1.set_title('Input window frequency spectrum')
                ax2.set_title('Output window frequency spectrum')

                plt.draw()

    # If considered as a silent window, just return original signal and reset phase coherency array to 1.0
    else:
        out = x
        Z = 1.0+0.0j
        print('Silence                   ', end='\r')

    return out, Z


def shift_factor(y, freq, notes, notes_name):
    """ Detect the fundamental frequency and compute the shift factor. Also computes the peaks position in freq spectrum
    :param y: Frequency spectrum (get via np.rfft)
    :param freq: Frequency corresponding to each index (get via np.rfftfreq)
    :param notes: notes table containing all notes (defined at beginning of code)
    :param notes_name: table containing all notes name (eg A#2) (defined at beginning of code)
    :return: shift factor, peaks indexes array, pitch (frequency of fundamental notes in window)
    """
    range_min = np.argmin(np.abs(freq - notes[0]))
    range_max = np.argmin(np.abs(freq - notes[-1]))
    peaks_idx, peaks_prop = find_peaks(np.abs(y[range_min:]),
                                       distance=80 / (freq[1] - freq[0]),
                                       prominence=60000)

    # If peaks are detected, fundamental freq is lowest peak
    if peaks_idx.shape[0] > 0 and peaks_idx[0] < range_max-range_min:
        peaks_idx = peaks_idx + range_min
        idx_fund = peaks_idx[0]

    # If no peak detected in range given by notes,
    # take highest frequency in spectrum (a peak but not detected by find_peaks)
    else:
        idx_fund = np.argmax(y[range_min:range_max]*np.conj(y[range_min:range_max])) + range_min

    # Interpolate with parabola between idx_fund-1 to idx_fund + 1
    polyfit = np.polyfit(freq[idx_fund - 1: idx_fund + 2],
                         np.abs(y)[idx_fund - 1:idx_fund + 2], 2)

    # Get max of parabola
    pitch = - polyfit[1] / (2 * polyfit[0])

    # Get closest note in key
    closest_note_idx = np.argmin(np.abs(pitch-notes))
    closest_note = notes[closest_note_idx]

    print("Closest note: ", repr(notes_name[closest_note_idx]), end=' \r')

    # Computation of shift factor: coeff to apply to frequency of input signal
    shift_f = closest_note / pitch

    # If shift f too large or to low (for security, shouldn't happen)
    if shift_f > 1.2 or shift_f<0.8:
        shift_f = 1
        print('Shift factor out of range, setting it to 1.0', flush=True)

    return shift_f, peaks_idx, pitch


def shift_freq(y, freq, shift_f):
    """
    Returns shifted frequency spectrum
    :param y: Fourier transform of window signal
    :param freq: array with corresponding frequency of Fourier transform for each idx
    :param shift_f: shift factor with which we shift the signal
    :return: y_new: shifted Fourier transform
    """
    # Interpolation: suppose you have correct pitch (freq_x = shift_f * freq) and resample to normal freq scale
    # to then do inverse fourier transform
    y_new = interp1d_p(shift_f * freq, y, freq, 'cubic')

    return y_new


def play(stream, chunk):
    """
    Write chunk to output stream
    :param stream: pyaudio output stream to write chunk
    :param chunk: numpy array to write to ouput stream
    """
    stream.write(chunk)


def silence(x, silence_threshold):
    """
    Return boolean indicating if the window is silent (if max of abs(signal) is below the threshold)
    :param x: window signal
    :param silence_threshold: threshold
    :return: boolean, silent (True) or not Silent (False)
    """
    amp_max = np.max(np.abs(x))
    return amp_max < silence_threshold


def window(w_size, overlap=0.5, type='sine'):
    """
    Return array of size w_size with window function evaluated at each index
    :param w_size: size of window
    :param overlap: 0, 0.5 or 0.75 of overlap
    :param type: type of the windows (sine, hann, hamming, or rect)
    :return: array containing window function evaluated between 0 and w_size-1
    """
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

    if type == 'sine':
        w = overlap_factor * np.sin((n+0.5)*np.pi/w_size)

    elif type == 'hann':
        w = overlap_factor * np.sin((n+0.5)*np.pi/w_size)**2

    elif type == 'rect':
        w = overlap_factor * np.ones(w_size)

    elif type == 'hamming':
        w = np.hamming(w_size)

    else:
        raise ValueError('Not valid window type')

    return w


def build_notes_vector(key, n_oct=4):
    """
    Construct two arrays containing frequency and notes names of notes present in specified key
    :param key: Name of key, has the form 'A' or 'Ab' or 'A#', letters from A to G
    :param n_oct: Number of octave to consider (default 4)
    :return: notes: array containing all frequency of notes in key
            notes_str_ex: array containing all notes names of notes in key
    """
    if key == 'chromatic':
        notes_str = np.asarray(['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'], dtype='<U3')
        notes_str_ex = np.tile(notes_str, (1, n_oct + 1))[0]
        octave_index = np.asarray([1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        for idx in range(len(notes_str_ex)):
            incr = idx//12
            notes_str_ex[idx] = str(notes_str_ex[idx]) + str(octave_index[idx%12]+incr)
        n_extended = np.arange(len(notes_str_ex))
        notes = np.asarray(55.0 * 2.0 ** (n_extended / 12.0))
        return notes, np.asarray(notes_str_ex)

    else:
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

    n_extended = np.array([])

    for i in range(n_oct + 1):
        n_extended = np.concatenate((n_extended, 12 * i + key_notes), axis=0)

    all_notes_str_ex = []

    for i in range(n_oct + 1):
        all_notes_str_ex = all_notes_str_ex + [n for n in notes_str]

    for i in range(len(all_notes_str_ex)):
        if i < 3:
            all_notes_str_ex[i] = all_notes_str_ex[i] + str(1)
        else:
            all_notes_str_ex[i] = all_notes_str_ex[i] + str(int((i-3)/12+2))

    all_notes_str_ex = np.asarray(all_notes_str_ex)
    notes_str_ex = all_notes_str_ex[n_extended.astype(np.int8)]
    notes_str_ex = np.asarray(notes_str_ex)

    # Notes for our table of notes, starting at 55Hz (A1)
    # The factor between each semitone is 2^(1/12)
    notes = np.asarray(55.0 * 2.0 ** (n_extended / 12.0))
    return notes, notes_str_ex


def C2P(x):
    """
    Take a complex np array and return a tuple of np array with norm and phase (polar coordinates)
    :param x: complex np array
    :return: tuple with norm array and phase array
    """
    return np.abs(x), np.angle(x)
