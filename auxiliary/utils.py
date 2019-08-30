from __future__ import print_function
import numpy as np
import pickle
import visdom
import torch
import sys
import os
import random

class COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def display_opts(opts):


    display_msg = """PARAMETERS:
            model         %s %s %s
            adjust        %s %s %s
            dataset       %s %s %s
            loadmodel     %s %s %s
            npatch        %s %s %s
            npoint        %s %s %s
            nlatent       %s %s %s
            nbatch        %s %s %s
            lrate         %s %s %s
            nepoch        %s %s %s
            first decay   %s %s %s
            second decay  %s %s %s
            training_id   %s %s %s
            """%(COLORS.OKGREEN,opts.model,COLORS.ENDC,
                 COLORS.OKGREEN,opts.adjust,COLORS.ENDC,
                 COLORS.OKGREEN,opts.dataset,COLORS.ENDC,
                 COLORS.OKGREEN,opts.loadmodel,COLORS.ENDC,
                 COLORS.OKGREEN,opts.npatch,COLORS.ENDC,
                 COLORS.OKGREEN,opts.npoint,COLORS.ENDC,
                 COLORS.OKGREEN,opts.nlatent,COLORS.ENDC,
                 COLORS.OKGREEN,opts.nbatch,COLORS.ENDC,
                 COLORS.OKGREEN,opts.lrate,COLORS.ENDC,
                 COLORS.OKGREEN,opts.nepoch,COLORS.ENDC,
                 COLORS.OKGREEN,opts.firstdecay,COLORS.ENDC,
                 COLORS.OKGREEN,opts.seconddecay,COLORS.ENDC,
                 COLORS.OKGREEN,opts.training_id,COLORS.ENDC)

    print(display_msg)


def display_it(mode, opt, epoch_id, batch_id, loss=None):
    """display iteration"""

    if batch_id % 50 == 0:
        msg = ''

        if mode == 'train':
            msg = "[%s%s%s] - %d/%d - %04d   %s%f%s" % (COLORS.OKGREEN,
                                                        opt.training_id,
                                                        COLORS.ENDC,
                                                        epoch_id,
                                                        opt.nepoch,
                                                        batch_id,
                                                        COLORS.BOLD,
                                                        loss,
                                                        COLORS.ENDC)

        if mode == 'valid':
            msg = "[%s%s%s] - %d/%d - %04d   %s%f%s" % (COLORS.OKBLUE,
                                                        opt.training_id,
                                                        COLORS.ENDC,
                                                        epoch_id,
                                                        opt.nepoch,
                                                        batch_id,
                                                        COLORS.BOLD,
                                                        loss,
                                                        COLORS.ENDC)

        if mode == 'test':
            msg = "[%s%s%s] - %d/%d" % (COLORS.WARNING,
                                        opt.training_id, COLORS.ENDC,
                                        epoch_id, opt.nepoch)
        print(msg)


class LOGGER:
    """logger of the network loss """

    def __init__(self):
        self.history = []
        self.data = []

    def add(self, val):
        self.data.append(val)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.history, f)

    def mean(self):
        m = np.mean(np.array(self.data))
        return m

    def reset(self):
        if self.data:
            self.history.append(np.mean(np.array(self.data)))
            self.data = []


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

CHUNK_SIZE = 150
lenght_line = 60
def my_get_n_random_lines(path, n=5):
    MY_CHUNK_SIZE = lenght_line * (n+2)
    lenght = os.stat(path).st_size
    with open(path, 'r') as file:
            file.seek(random.randint(400, lenght - MY_CHUNK_SIZE))
            chunk = file.read(MY_CHUNK_SIZE)
            lines = chunk.split(os.linesep)
            return lines[1:n+1]
