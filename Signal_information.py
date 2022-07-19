from Lightpath import Lightpath

'''class Lightpath(object):
    def __init__(self, power, path, channel):
        self._signal_power = power
        self._path = path
        self._channel = channel
        self._noise_power = 0
        self._latency = 0
        self.Rs = 32e9
        self.df = 50e9

    @property
    def signal_power(self):
        return self._signal_power

    def set_signal_power(self, power):
        self._signal_power = power

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def channel(self):
        return self._channel

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise_power):
        self._noise_power = noise_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    def add_noise(self, noise):
        self.noise_power += noise

    def add_latency(self, latency):
        self.latency += latency

    def next(self):
        self._path = self.path[1:]'''


class SignalInformation(Lightpath):

    def __init__(self, signal_power, path):
        super().__init__(signal_power, path, 0)
        self._signal_power = signal_power
        self._noise_power = 0
        self._latency = 0
        self._path = path
        self.Rs = 32e9
        self.df = 50e9

    def increase_noise(self, noise_power):
        self._noise_power += noise_power

    def increase_latency(self, latency):
        self._latency += latency

    def next(self):
        self._path = self._path[1:]

    @property
    def path(self):
        return self._path

    def getlatency(self):
        return self._latency

    def getnoisepower(self):
        return self._noise_power

    def getsignalpower(self):
        return self._signal_power
