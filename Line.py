import numpy as np
from scipy.constants import c, h, pi


class Line(object):
    def __init__(self, line_dict):
        self._label = line_dict["label"]
        self._length = line_dict["length"]
        self._successive = {}  # dict [Node]
        self._state = ["free"] * 10
        self._n_amplifiers = int(self._length / 80e3)  # one amp every 80km
        self._gain = 16
        self._noise_figure = 3  # 5#
        self._alpha = 0.2 * 0.001  # fiber loss dB/m
        self._beta2 = 2.3e-26  # 0.6e-26  #
        self._gamma = 1.27e-3
        self._Rs = 32e9
        self._df = 50e9

    @property
    def gain(self):
        return self._gain

    @property
    def noise_figure(self):
        return self._noise_figure

    @property
    def n_amplifiers(self):
        return self._n_amplifiers

    @n_amplifiers.setter
    def n_amplifiers(self, amp):
        self._n_amplifiers = amp

    @property
    def state(self):
        return self._state

    def free_state(self):
        self._state = ["free"] * 10

    @state.setter
    def state(self, state):
        state = [s.strip().lower() for s in state]  # no spaces and all in lower c.
        if set(state).issubset({"free", "occupied"}):
            self._state = state
        else:
            print("ERROR: No state value:", set(state) - {"free", "occupied"})

    def latency_generation(self):
        latency = self._length / (c * 2 / 3)
        return latency

    def noise_generation(self, lightpath):
        # noise = 0.000000001 * signal_power * self._length  # 1e-9 * s_p * length
        noise = self.ase_generation() + self.nli_generation(lightpath.signal_power, lightpath.df, lightpath.Rs)

        return noise

    def propagate(self, lightpath, occupation=False):
        latency = self.latency_generation()
        lightpath.add_latency(latency)
        sp = self.optimized_launch_power(self.eta_nli(lightpath.df, lightpath.Rs))
        lightpath.set_signal_power(sp)
        noise = self.noise_generation(lightpath)
        lightpath.add_noise(noise)
        if occupation:
            channel = lightpath.channel
            new_state = self._state.copy()
            new_state[channel] = "occupied"
            self._state = new_state
        node = self._successive[lightpath.path[0]]
        lightpath = node.propagate(lightpath, occupation)
        return lightpath

    def ase_generation(self):
        NF = 10 ** (self._noise_figure / 10)
        G = 10 ** (self._gain / 10)
        f = 193.414e12
        Bn = 12.5e9  # GHz
        ASE = self.n_amplifiers * h * f * Bn * NF * (G - 1)
        return ASE

    def nli_generation(self, signal_power, dfp, Rsp):
        Bn = 12.5e9  # GHz
        eta_nli = self.eta_nli(dfp, Rsp)
        nli = (signal_power ** 3) * eta_nli * self._n_amplifiers * Bn
        return nli

    def eta_nli(self, dfp, Rsp):
        df = dfp
        Rs = Rsp
        a = self._alpha / (20 * np.log10(np.e))
        Nch = 10
        b2 = self._beta2
        e_nli = 16 / (27 * np.pi) * np.log(
            np.pi ** 2 * b2 * Rs ** 2 * Nch ** (2 * Rs / df) / (2 * a)) * self._gamma ** 2 / (4 * a * b2 * Rs ** 3)
        return e_nli

    def optimized_launch_power(self, eta):
        F = 10 ** (self.noise_figure / 10)
        G = 10 ** (self.gain / 10)
        f0 = 193.414e12
        olp = ((F * f0 * h * G) / (2 * eta)) ** (1 / 3)
        return olp