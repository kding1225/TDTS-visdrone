# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import json
from collections import defaultdict
from collections import deque
import torch


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, deque_data=None, series=None, total=0.0, count=0):
        self.deque = deque(deque_data, maxlen=window_size) if deque_data else deque(maxlen=window_size)
        self.series = series or []
        self.total = total
        self.count = count
        self.window_size = window_size

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    def to_dict(self):
        data = dict(
            deque_data=list(self.deque),
            series=self.series,
            total=self.total,
            count=self.count,
            window_size=self.window_size
        )
        return data


class MetricLogger(object):
    def __init__(self, delimiter="\t", save_dir=""):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.save_dir = save_dir

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def save(self, file_path="", is_main_process=True):
        """
        save current state to json file
        """
        if is_main_process:
            try:
                file_path = file_path or self.save_dir
                data = dict(delimiter=self.delimiter)
                for name, meter in self.meters.items():
                    data[name] = meter.to_dict()

                with open(file_path, 'w') as f:
                    json.dump(data, f)
            except:
                print('did not save')

    def load(self, file_path="", is_main_process=True):
        """
        recover the state from a json file
        """
        if is_main_process:
            try:
                file_path = file_path or self.save_dir
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    self.delimiter = data.pop('delimiter')
                    for name, meter_data in data.items():
                        # print(name, meter_data)
                        self.meters[name] = SmoothedValue(**meter_data)
            except:
                print('did not load')
