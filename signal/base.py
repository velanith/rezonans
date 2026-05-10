from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

fs = 10e6 # sampling frequency
t = np.linspace(0, 1e-4, int(fs * 1e-4), endpoint=False)
f = 200e3

sine = np.cos(2 * np.pi * f * t)

freqs, psd = signal.periodogram(sine, fs)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(t * 1e6, sine)
plt.xlabel("Zaman (mikrosaniye)")
plt.title("200 kHz Sinüs Dalgası")

plt.subplot(1,2,2)
plt.semilogy(freqs / 1e3, psd)
plt.xlabel("Frekans (kHz)")
plt.xlim(0, 500)
plt.title("Güç spektral yoğunluğu")

plt.tight_layout()
plt.show()


