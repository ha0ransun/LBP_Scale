from abc import ABC, abstractmethod

class AbstractRateStats(ABC):

    @abstractmethod
    def step(self, average_rate, num_samples):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def get(self):
        raise NotImplementedError


class AlltimeRateStats(AbstractRateStats):
    def __init__(self):
        super(AlltimeRateStats, self).__init__()
        self.reset()

    def step(self, average_rate, num_samples):
        self.stats_acc += average_rate * num_samples
        self.num += num_samples

    def reset(self):
        self.stats_acc = 0.0
        self.num = 0.0

    def get(self):
        if self.num == 0:
            return 0.0
        return self.stats_acc / self.num


class ExpAvgRateStats(AbstractRateStats):
    def __init__(self, momentum=0.9):
        super(ExpAvgRateStats, self).__init__()
        self.momentum = momentum
        self.reset()

    def step(self, average_rate, num_samples):
        if self.hist_rate is None:
            self.hist_rate = average_rate
        else:
            self.hist_rate = self.momentum * self.hist_rate + (1 - self.momentum) * average_rate
    
    def reset(self):
        self.hist_rate = None
    
    def get(self):
        if self.hist_rate is None:
            return 0.0
        return self.hist_rate


class WindowAvgRateStats(AbstractRateStats):
    def __init__(self, window_size=10):
        super(WindowAvgRateStats, self).__init__()
        self.window_size = window_size
        self.reset()
    
    def step(self, average_rate, num_samples):
        if len(self.window) < self.window_size:
            self.window.append((average_rate, num_samples))
        else:
            self.window[self.pos] = (average_rate, num_samples)
            self.pos = (self.pos + 1) % self.window_size

    def reset(self):
        self.pos = 0
        self.window = []

    def get(self):
        tot_rate = sum([x[0] * x[1] for x in self.window])
        tot_num = sum([x[1] for x in self.window])
        if tot_num == 0:
            return 0.0
        return tot_rate / tot_num


def get_rate_stats(rs_type):
    if rs_type.startswith('exp'):
        m = float(rs_type.split('-')[-1])
        return ExpAvgRateStats(m)
    elif rs_type.startswith('window'):
        w = int(rs_type.split('-')[-1])
        return WindowAvgRateStats(w)
    elif rs_type == 'all':
        return AlltimeRateStats()
    else:
        raise ValueError('Unknown rate stats type %s' % rs_type)
