class Connection(object):
    def __init__(self, in_node, out_node, s_power):
        self._in_node = in_node
        self._out_node = out_node
        self._sig_power = s_power
        self._latency = 0
        self._snr = 0
        self._rb = 0

    @property
    def bit_rate(self):
        return self._rb

    @bit_rate.setter
    def bit_rate(self, rb):
        self._rb = rb

    @property
    def in_node(self):
        return self._in_node

    @property
    def out_node(self):
        return self._out_node

    @property
    def signal_power(self):
        return self._sig_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, snr):
        self._snr = snr
