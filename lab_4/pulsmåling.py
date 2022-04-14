import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


def getMeasurementMatrix(fileName):
    """
    Gets measurement vetors in matrix form: [R-vec, G-vec, B-vec]

    :param fileName: csv-file with data
    :return: measurement matrix
    """
    dataRed = []
    dataGreen = []
    dataBlue = []

    with open(fileName) as f:
        # reading each line
        for line in f:
            line_split = line.split()
            dataRed.append(float(line_split[0]))
            dataGreen.append(float(line_split[1]))
            dataBlue.append(float(line_split[2]))
    return [dataRed, dataGreen, dataBlue]


def detrendMeasurementMatrix(matrix):
    """
    Detrends vectors in measurement matrix

    :param matrix: measurement matrix
    :return: detrended matrix
    """
    res_matrix = []
    for vec in matrix:
        res_matrix.append(signal.detrend(vec, axis=0))
    return res_matrix


def filterMeasurementMatrix(matrix, N, W_n, filter_type, fs=40):
    """
    Filters vectors with Butterworth-filter
    
    :param N: order of filter
    :param matrix: measurement matrix
    :param filter_type: type of filter(‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’)
    :param W_n: critical frequencies, scalar for low/high-pass, len-2 vec for bandpass/-stop 
    :param fs: sampling frequency(def = 40)
    :return: filtered matrix
    """
    filter = signal.butter(N=N, Wn=W_n, btype=filter_type, output="sos", fs=fs)
    res_matrix = []
    for vec in matrix:
        res_matrix.append(signal.sosfilt(filter, vec))
    return res_matrix


def getPulse(matrix, fs=40):
    """
    Returns estimated pulse for all channels in measurement matrix

    :param matrix: measurement matrix
    :param fs: sampling frequency
    :return: pulse vector with estimated pulses for all channels
    """
    res_vec = []
    for vec in matrix:
        freq = np.fft.fftfreq(n=len(vec), d=1 / fs)
        spectrum = abs(np.fft.fft(vec, n=len(vec)))
        samp_freq = np.argmax(spectrum)
        res_vec.append(np.abs(freq[samp_freq]))
    return res_vec


def getPulses(matrices):
    """
    Returns estimated pulse for all channels and matrices in matrix vector

    :param matrices: vector of measurement matricies
    :return: vector of pulse vectors with estimated pulses for all channels
    """
    res_matrix = []
    for matrix in matrices:
        res_matrix.append(getPulse(matrix))
    return res_matrix


def getMeanForAllChannels(pulse_matrix):
    """
    Gets mean of all channels for for each pulse measurement vector

    :param pulse_matrix: matrix with pulses
    :return: average for each pulse measurement vector
    """
    res_vec = []
    for i in range(len(pulse_matrix)):
        res_vec.append(np.mean(pulse_matrix[i]))
    return res_vec


def getVarOfChannels(pulse_matrix, real_pulse_vec):
    """
    Gets variance for all measurements for each channel

    :param pulse_matrix: matrix with pulses
    :param real_pulse_vec: vector with real value for pulse
    :return: vector with variance for each channel
    """
    res_vec = []
    if (len(pulse_matrix) == len(real_pulse_vec)):
        pulse_matrix = np.transpose(pulse_matrix)
        for i in range(len(pulse_matrix)):
            pulse_vec = pulse_matrix[i]
            N = len(pulse_vec)
            res_vec.append(np.sum(np.power(real_pulse_vec - pulse_matrix[i], 2)) / (N - 1))
        return res_vec
    else:
        print("Non-matching vector lengths")


def getPulseMeasurements(file_vec, detrend=True, filter=True):
    """
    Gets pulse measurments from files in file_vec

    :param file_vec: vector with file names
    :param detrend: detrend setting True=detrend, False=not detrend (default: True)
    :param detrend: filter setting True=filter, False=no filter (default: True)
    :return: vector of pulse vectors with estimated pulses for all channels
    """
    matrices = []
    for file in file_vec:
        matrix = getMeasurementMatrix(file)
        if detrend:
            matrix = detrendMeasurementMatrix(matrix)
            if filter:
                matrix = filterMeasurementMatrix(matrix, N=5, W_n=[0.5, 3.5], filter_type="bandpass", fs=40)
        matrices.append(matrix)
    return getPulses(matrices)


def plotTimeDomainSignals(matrix, fs, labels, colors):
    """
    Plots signals in matrix in time-domain

    :param matrix: matrix with signals
    :param fs: sample frequency
    :param labels: labels for legend
    :param colors: colors of signal plots
    """
    matrix = np.array(matrix)
    if matrix.ndim == 2:
        for vec, l, c in zip(matrix, labels, colors):
            print(c)
            x = np.arange(0, len(vec) / fs, 1 / fs)
            plt.plot(x, vec, c=c, label=l)
        plt.xlabel("Sekunder[s]")
        plt.legend()
        plt.show()
    else:
        raise Exception("Matrix has wrong dimensions. Should be 2-dimensional, but is:",
                        matrix.ndim, "dimensional")


def plotFrequencyDomainSignals(matrix, fs, labels, colors):
    """
    Plots signals in matrix in frequency-domain

    :param matrix: matrix with signals
    :param fs: sample frequency
    :param labels: labels for legend
    :param colors: colors of signal plots
    """
    matrix = np.array(matrix)
    if matrix.ndim == 2:
        for vec, l, c in zip(matrix, labels, colors):
            freq = np.fft.fftfreq(n=len(vec), d=1 / fs)
            spectrum = abs(np.fft.fft(vec, n=len(vec)))
            plt.plot(freq, spectrum, c=c, label=l)
        plt.xlabel("Frekvens[Hz]")
        plt.legend()
        plt.show()
    else:
        raise Exception("Matrix has wrong dimensions. Should be 2-dimensional, but is:",
                        matrix.ndim, "dimensional")


def main():
    # Get measurments
    matrix = getMeasurementMatrix("brage_hvil_trans_3.txt")
    # Detrend data
    detrended_matrix = detrendMeasurementMatrix(matrix)
    # Filter data
    filtered_matrix = filterMeasurementMatrix(detrended_matrix, N=20,
                                              W_n=[0.5, 3.5], filter_type="bandpass", fs=40)

    # Plot
    plotTimeDomainSignals(filtered_matrix[0:1], fs=40, labels=["Red"], colors=['r'])
    plotFrequencyDomainSignals(filtered_matrix[0:1], fs=40, labels=["Red"], colors=['r'])

    file_vec = [f"brage_hvil_trans_{i}.txt" for i in range(1, 5 + 1)]
    pulse_matrix = getPulseMeasurements(file_vec, detrend=True,
                                        filter=True)  # Matrise med pulse for all målinger og kanaler

    # Using pulse clock meassuremnt as reference
    pulse_clock_measurement_vec = np.array([84, 80, 80, 76, 76]) / 60
    means = getMeanForAllChannels(pulse_matrix)
    print("Puls klokke:", pulse_clock_measurement_vec)
    print("Gjennomsnitt:", means)

    vars = getVarOfChannels(pulse_matrix, pulse_clock_measurement_vec)
    print("Vars", vars)


if __name__ == '__main__':
    main()
