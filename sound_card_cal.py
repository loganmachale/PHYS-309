# Library imports, data import function definition
import numpy as np
import plotly.graph_objects as go
from scipy.io import wavfile
from scipy.optimize import curve_fit
from scipy.interpolate import Rbf

def importWAV(filename):
    samplerate, rawData = wavfile.read(filename)
    
    time = np.linspace(0, rawData.shape[0]/samplerate, rawData.shape[0])   
    
    data = {'left':rawData[:, 0],'right':rawData[:, 1]}
    return time,data

# SC Data imports
v_100 = importWAV('Calibration Signals/40mV_100Hz.wav')
v_500 = importWAV('Calibration Signals/40mV_500Hz.wav')
v_1k = importWAV('Calibration Signals/40mV_1kHz.wav')
v_2k = importWAV('Calibration Signals/40mV_2kHz.wav')
v_4k = importWAV('Calibration Signals/40mV_4kHz.wav')
v_6k = importWAV('Calibration Signals/40mV_6kHz.wav')
v_8k = importWAV('Calibration Signals/40mV_8kHz.wav')
v_11k = importWAV('Calibration Signals/40mV_11kHz.wav')
v_13k = importWAV('Calibration Signals/40mV_13kHz.wav')
v_15k = importWAV('Calibration Signals/40mV_15kHz.wav')
v_18k = importWAV('Calibration Signals/40mV_18kHz.wav')

t_100 = v_100[0]
t_500 = v_500[0]
t_1k = v_1k[0]
t_2k = v_2k[0]
t_4k = v_4k[0]
t_6k = v_6k[0]
t_8k = v_8k[0]
t_11k = v_11k[0]
t_13k = v_13k[0]
t_15k = v_15k[0]
t_18k = v_18k[0]

t_arr = np.array([t_100, t_500, t_1k, t_2k, t_4k, t_6k, t_8k, t_11k, t_13k, t_15k, t_18k])

v_100 = v_100[1]['right']
v_500 = v_500[1]['right']
v_1k = v_1k[1]['right']
v_2k = v_2k[1]['right']
v_4k = v_4k[1]['right']
v_6k = v_6k[1]['right']
v_8k = v_8k[1]['right']
v_11k = v_11k[1]['right']
v_13k = v_13k[1]['right']
v_15k = v_15k[1]['right']
v_18k = v_18k[1]['right']

v_arr = np.array([v_100, v_500, v_1k, v_2k, v_4k, v_6k, v_8k, v_11k, v_13k, v_15k, v_18k])

sc_f_arr = np.array([1E2, 5E2, 1E3, 2E3, 4E3, 6E3, 8E3, 11E3, 13E3, 15E3, 18E3])

# Calculating calibration coefficients for SC data
V_amp_arr = np.sqrt(2) * np.array([29.24E-3, 28.25E-3, 28.25E-3, 28.25E-3, 28.25E-3, 29.23E-3,
                                   28.25E-3, 29.3E-3 ,29.19E-3, 29.19E-3, 29.18E-3])
V_amp_std = 0.02E-3

C_arr = np.array([])
C_std_arr = np.array([])

for v, f, V_amp in zip(v_arr, sc_f_arr, V_amp_arr):
    n = round(2 * f)
    v_c = np.copy(v)
    v_sorted = np.sort(v_c)
    amps = (v_sorted[(-n - 1): -51] - v_sorted[50: n]) / 2
    C = np.mean(amps) / V_amp
    C_std = 1 / V_amp * np.sqrt((np.std(amps, ddof=1)) ** 2 + (np.mean(amps) * V_amp_std / V_amp) ** 2)
    
    C_arr = np.append(C_arr, C)
    C_std_arr = np.append(C_std_arr, C_std)

# Generating interpolation functions for SC data
def O4_T_series(x, a, b, c, d, e, f, g):
    return a + b*(x+c)**2 + d*(x+e)**3 + f*(x+g)**4
p2, p2_cov = curve_fit(O4_T_series, sc_f_arr, C_arr, sigma=C_std_arr, maxfev=10000)


rbf_interpolator = Rbf(sc_f_arr, C_arr, function='inverse_multiquadric')


def sc_cal_coeff(f):
    return 0.7 * O4_T_series(f, *p2) + 0.3 * rbf_interpolator(f)