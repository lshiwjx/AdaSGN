# import torchvision
import torch
import numpy as np
# import matplotlib.pyplot as plt
import time


class Logger:
    """
    A logger
    """

    def __init__(self, title):
        print("{}".format(title))
        self.addr = None

    def log(self, string):
        current_time = time.asctime(time.localtime(time.time()))[4:]  # print current time
        s = "[{}] {}".format(current_time, string)  # remove week information
        print(s)
        fid = open(self.addr, 'a')
        fid.write("%s\n" % s)
        fid.close()


class IteratorTimer:
    """
    An iterator to produce duration. self.last_duration
    """

    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = self.iterable.__iter__()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.iterable)

    def __next__(self):
        start = time.time()
        n = self.iterator.__next__()
        self.last_duration = (time.time() - start)
        return n

    next = __next__


class Recorder:
    """
    record the avg, best
    """
    def __init__(self, larger_is_better=True, previous_items=0):
        self.larger_is_better = larger_is_better
        self.best_at = None
        self.best_val = None
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = previous_items

    def is_better_than(self, x, y):
        if self.larger_is_better:
            return x > y
        else:
            return x < y

    def update(self, val, n=1):
        if self.count == 0 or self.is_better_than(val, self.best_val):
            self.best_val = val
            self.best_at = self.count
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def is_current_best(self, n=1):
        return self.count == self.best_at + n

