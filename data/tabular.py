import os
import torch as to
import torch.autograd as ag
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from amb.data.profiler import DataProfiler
from amb.settings import settings
import sys
sys.path.append(os.path.join(settings.ROOT_DIR, 'tests'))
from datasets import Datasets

home = os.path.expanduser('~')

import logging


def make_batches(size, batch_size, include_partial=False):
    """Returns a list of batch indices (tuples of indices).
    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.
    # Returns
        A list of tuples of array indices.
    """
    if include_partial:
        func = np.ceil
    else:
        func = np.floor
    num_batches = int(func(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)]


def reshape_ts(seq, tau, pad=False, pad_val=np.nan):
    """Reshape a T x N time-series into a T x tau X N tensor, where tau is
    the length of the lookback window.
    """
    N = len(seq) - tau + 1
    nf = seq.shape[1]
    new_seq = np.ndarray((N, tau, nf))
    for i in range(0, N):
        new_seq[i, :, :] = seq[i:i + tau]
    if pad:
        pad = np.empty((tau - 1, tau, nf))
        pad[:] = pad_val
        new_seq = np.concatenate((pad, new_seq))
    return new_seq


class Tabular(object):
    def __init__(self, args):

        if args.dataset == 'beijing':
            f_train = os.path.join(home, 'amb-data/beijing_train.h5')
            f_test = os.path.join(home, 'amb-data/beijing_test.h5')
            target = 'pm2.5'

            df_train = pd.read_hdf(f_train)
            df_test = pd.read_hdf(f_test)

        elif args.dataset == 'simple':
            # simpleRNN gets 100% accuracy after ~25 epochs with SGD(lr=0.01)
            d = Datasets.SimpleRecurrent.make(n_samples=5000, n_features=10, n_informative=1, n_classes=0)
            target = d.meta().target
            df_train = d.train()
            df_test = d.test()

        elif args.dataset == 'brenda':
            f_train = os.path.join(home, 'amb-data/brenda_reg_train.h5')
            f_test = os.path.join(home, 'amb-data/brenda_reg_test.h5')
            # target = 'subsea_barrier_temperature'
            target = 'flow_rate'

            df_train = pd.read_hdf(f_train)
            df_test = pd.read_hdf(f_test)

        logging.info('n_train: %i, n_test: %i' % (df_train.shape[0], df_test.shape[0]))

        dp = DataProfiler(target=target)
        x_train, y_train = dp.fit_transform(df_train)
        x_test, y_test = dp.transform(df_test)

        is_class = dp.has_categorical_target()
        n_inputs = x_train.shape[1]
        n_outputs = dp.num_target_classes()
        if n_outputs == 0:
            n_outputs = 1
        # n_hidden = n_inputs  # needs to be same as number of features for now

        # seq_len = shared_rnn_max_length
        # x_train = reshape_ts(x_train.values, seq_len)
        # y_train = y_train.iloc[seq_len - 1:].values

        ntrain = int(np.round(x_train.shape[0] * 0.7))
        idx = np.arange(x_train.shape[0])
        idx_train, idx_val = idx[:ntrain], idx[ntrain:]

        idx_train, idx_val = to.LongTensor(idx_train), to.LongTensor(idx_val)

        X = x_train.values
        y = y_train.values

        X = to.Tensor(X.astype('float32'))
        if is_class:
            y = y.dot(np.arange(y.shape[1])).astype(int)
            y = to.LongTensor(y)
        else:
            y = to.Tensor(y.astype('float32'))
        X = ag.Variable(X)
        y = ag.Variable(y)

        kwargs = {'num_workers': 0, 'pin_memory': False}

        # train_loader = DataLoader(TensorDataset(X[idx_train].data, y[idx_train].data),
        #     batch_size=args.batch_size, shuffle=False, **kwargs)

        # test_loader = DataLoader(TensorDataset(X[idx_val].data, y[idx_val].data),
        #     batch_size=args.batch_size, shuffle=False, **kwargs)

        # self.train = train_loader
        # self.valid = test_loader
        # self.test = test_loader

        self.train = TensorDataset(X[idx_train].data, y[idx_train].data)
        self.valid = TensorDataset(X[idx_val].data, y[idx_val].data)

        args.n_inputs = n_inputs
        args.n_outputs = n_outputs
        args.recurrent = dp.is_timeseries
        
        # self.train = t.utils.data.DataLoader(
        #     Dataset(root='./data', train=True, transform=transform, download=True),
        #     batch_size=args.batch_size, shuffle=True,
        #     num_workers=2, pin_memory=True)

        # self.valid = t.utils.data.DataLoader(
        #     Dataset(root='./data', train=False, transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         normalize,
        #     ])),
        #     batch_size=args.batch_size, shuffle=False,
        #     num_workers=2, pin_memory=True)

        # self.test = self.valid
