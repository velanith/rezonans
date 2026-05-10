import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Literal
from signal.coil import CoilCollection


class Visualizer:

    def __init__(self, coil_collection: CoilCollection):
        self.cc = coil_collection

    def _time_axis(self, n_cycles: int = 5, n_points: int = 1000):
        T = 1.0 / self.cc.frequency
        return np.linspace(0, n_cycles * T, n_points)

    def _signal(self, coil_idx: int, t: np.ndarray,
                waveform: Literal["sine", "square"] = "sine") -> np.ndarray:
        coil = self.cc.coils[coil_idx]
        angle = self.cc.omega * t + coil.phase
        if waveform == "sine":
            return coil.current * np.sin(angle)
        elif waveform == "square":
            return coil.current * np.sign(np.sin(angle))

    def plot_single(self, coil_idx: int = 0,
                    waveform: Literal["sine", "square"] = "sine"):
        t = self._time_axis()
        s = self._signal(coil_idx, t, waveform)
        coil = self.cc.coils[coil_idx]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t * 1e6, y=s, mode="lines",
            name=f"Coil {coil_idx}",
            line=dict(width=2)
        ))
        fig.update_layout(
            title=f"Coil {coil_idx} — {waveform} — f={self.cc.frequency/1e3:.0f} kHz, "
                  f"I={coil.current} A, φ={np.degrees(coil.phase):.1f}°",
            xaxis_title="Zaman (µs)",
            yaxis_title="Akım (A)",
            template="plotly_dark",
            hovermode="x unified"
        )
        fig.show()

    def plot_all(self, waveform: Literal["sine", "square"] = "sine"):
        t = self._time_axis()
        fig = go.Figure()

        for i, coil in enumerate(self.cc.coils):
            s = self._signal(i, t, waveform)
            fig.add_trace(go.Scatter(
                x=t * 1e6, y=s, mode="lines",
                name=f"Coil {i} φ={np.degrees(coil.phase):.1f}°",
                line=dict(width=2)
            ))

        fig.update_layout(
            title=f"Tüm Coil'ler — {waveform} — f={self.cc.frequency/1e3:.0f} kHz",
            xaxis_title="Zaman (µs)",
            yaxis_title="Akım (A)",
            template="plotly_dark",
            hovermode="x unified"
        )
        fig.show()

    def plot_interference(self, waveform: Literal["sine", "square"] = "sine"):
        t = self._time_axis()
        superposition = sum(self._signal(i, t, waveform)
                            for i in range(len(self.cc.coils)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t * 1e6, y=superposition, mode="lines",
            name="Süperpozisyon",
            line=dict(color="crimson", width=2)
        ))
        fig.update_layout(
            title=f"Interference — {waveform} — Süperpozisyon",
            xaxis_title="Zaman (µs)",
            yaxis_title="Toplam Akım (A)",
            template="plotly_dark",
            hovermode="x unified"
        )
        fig.show()

    def plot_fft(self, waveform: Literal["sine", "square"] = "sine"):
        t = self._time_axis(n_cycles=20, n_points=4096)
        superposition = sum(self._signal(i, t, waveform)
                            for i in range(len(self.cc.coils)))

        dt = t[1] - t[0]
        freqs = np.fft.rfftfreq(len(t), d=dt)
        magnitude = np.abs(np.fft.rfft(superposition))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=freqs / 1e3, y=magnitude, mode="lines",
            name="FFT",
            line=dict(width=2)
        ))
        fig.update_layout(
            title=f"FFT — {waveform}",
            xaxis_title="Frekans (kHz)",
            yaxis_title="Genlik",
            xaxis_range=[0, self.cc.frequency / 1e3 * 10],
            template="plotly_dark",
            hovermode="x unified"
        )
        fig.show()

    def plot_waveform_comparison(self, coil_idx: int = 0):
        t = self._time_axis()
        sine = self._signal(coil_idx, t, "sine")
        square = self._signal(coil_idx, t, "square")

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Sinüs Dalga", "Kare Dalga"))

        fig.add_trace(go.Scatter(x=t * 1e6, y=sine, mode="lines",
                                 name="Sinüs", line=dict(color="steelblue", width=2)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=t * 1e6, y=square, mode="lines",
                                 name="Kare", line=dict(color="darkorange", width=2)),
                      row=1, col=2)

        fig.update_xaxes(title_text="Zaman (µs)")
        fig.update_yaxes(title_text="Akım (A)")
        fig.update_layout(
            title=f"Dalga Karşılaştırması — f={self.cc.frequency/1e3:.0f} kHz",
            template="plotly_dark"
        )
        fig.show()

    def plot_fft_comparison(self, coil_idx: int = 0):
        t = self._time_axis(n_cycles=20, n_points=4096)
        dt = t[1] - t[0]
        freqs = np.fft.rfftfreq(len(t), d=dt)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("FFT — Sinüs", "FFT — Kare"))

        colors = ["steelblue", "darkorange"]
        for col, (waveform, color) in enumerate(zip(["sine", "square"], colors), start=1):
            s = self._signal(coil_idx, t, waveform)
            magnitude = np.abs(np.fft.rfft(s))
            fig.add_trace(go.Scatter(
                x=freqs / 1e3, y=magnitude, mode="lines",
                name=f"FFT {waveform}", line=dict(color=color, width=2)),
                row=1, col=col)

        fig.update_xaxes(title_text="Frekans (kHz)",
                         range=[0, self.cc.frequency / 1e3 * 10])
        fig.update_yaxes(title_text="Genlik")
        fig.update_layout(
            title=f"FFT Karşılaştırması — f={self.cc.frequency/1e3:.0f} kHz",
            template="plotly_dark"
        )
        fig.show()