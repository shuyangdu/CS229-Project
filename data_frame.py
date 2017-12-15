import copy
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataFrame(object):

    """Minimal pd.DataFrame analog for handling n-dimensional numpy matrices with additional
    support for shuffling, batching, and train/test splitting.
    Args:
        columns: List of names corresponding to the matrices in data.
        data: List of n-dimensional data matrices ordered in correspondence with columns.
            All matrices must have the same leading dimension.  Data can also be fed a list of
            instances of np.memmap, in which case RAM usage can be limited to the size of a
            single batch.
    """

    def __init__(self, columns, data):
        assert len(columns) == len(data), 'columns length does not match data length'

        lengths = [mat.shape[0] for mat in data]
        assert len(set(lengths)) == 1, 'all matrices in data must have same first dimension'

        self.length = lengths[0]
        self.columns = columns
        self.data = data
        self.dict = dict(zip(self.columns, self.data))
        self.idx = np.arange(self.length)

    def shapes(self):
        return pd.Series(dict(zip(self.columns, [mat.shape for mat in self.data])))

    def dtypes(self):
        return pd.Series(dict(zip(self.columns, [mat.dtype for mat in self.data])))

    def shuffle(self):
        np.random.shuffle(self.idx)

    def train_test_split(self, train_size, random_state=np.random.randint(10000)):
        train_idx, test_idx = train_test_split(self.idx, train_size=train_size, random_state=random_state)
        train_df = DataFrame(copy.copy(self.columns), [mat[train_idx] for mat in self.data])
        test_df = DataFrame(copy.copy(self.columns), [mat[test_idx] for mat in self.data])
        return train_df, test_df

    def batch_generator(self, batch_size, shuffle=True, num_epochs=10000, allow_smaller_final_batch=False):
        epoch_num = 0
        while epoch_num < num_epochs:
            if shuffle:
                self.shuffle()

            for i in range(0, self.length + 1, batch_size):
                batch_idx = self.idx[i: i + batch_size]
                if not allow_smaller_final_batch and len(batch_idx) != batch_size:
                    break
                yield DataFrame(columns=copy.copy(self.columns), data=[mat[batch_idx].copy() for mat in self.data])

            epoch_num += 1

    def iterrows(self):
        for i in self.idx:
            yield self[i]

    def mask(self, mask):
        return DataFrame(copy.copy(self.columns), [mat[mask] for mat in self.data])

    def __iter__(self):
        return self.dict.items().__iter__()

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.dict[key]

        elif isinstance(key, int):
            return pd.Series(dict(zip(self.columns, [mat[self.idx[key]] for mat in self.data])))

    def __setitem__(self, key, value):
        assert value.shape[0] == len(self), 'matrix first dimension does not match'
        if key not in self.columns:
            self.columns.append(key)
            self.data.append(value)
        self.dict[key] = value


class DataReader(object):

    def __init__(self, data_dir):
        data_cols = [
            'data',
            'is_nan',
            'page_id',
            'project',
            'access',
            'agent',
            'test_data',
            'test_is_nan'
        ]
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i))) for i in data_cols]

        self.test_df = DataFrame(columns=data_cols, data=data)
        self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.95)

        print 'train size', len(self.train_df)
        print 'val size', len(self.val_df)
        print 'test size', len(self.test_df)

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    def test_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=True,
            num_epochs=1,
            is_test=True
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, is_test=False):
        batch_gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=is_test
        )
        data_col = 'test_data' if is_test else 'data'
        is_nan_col = 'test_is_nan' if is_test else 'is_nan'
        for batch in batch_gen:
            num_decode_steps = 64
            full_seq_len = batch[data_col].shape[1]
            max_encode_length = full_seq_len - num_decode_steps if not is_test else full_seq_len

            x_encode = np.zeros([len(batch), max_encode_length])
            y_decode = np.zeros([len(batch), num_decode_steps])
            is_nan_encode = np.zeros([len(batch), max_encode_length])
            is_nan_decode = np.zeros([len(batch), num_decode_steps])
            encode_len = np.zeros([len(batch)])
            decode_len = np.zeros([len(batch)])

            for i, (seq, nan_seq) in enumerate(zip(batch[data_col], batch[is_nan_col])):
                rand_len = np.random.randint(max_encode_length - 365 + 1, max_encode_length + 1)
                x_encode_len = max_encode_length if is_test else rand_len
                x_encode[i, :x_encode_len] = seq[:x_encode_len]
                is_nan_encode[i, :x_encode_len] = nan_seq[:x_encode_len]
                encode_len[i] = x_encode_len
                decode_len[i] = num_decode_steps
                if not is_test:
                    y_decode[i, :] = seq[x_encode_len: x_encode_len + num_decode_steps]
                    is_nan_decode[i, :] = nan_seq[x_encode_len: x_encode_len + num_decode_steps]

            batch['x_encode'] = x_encode
            batch['encode_len'] = encode_len
            batch['y_decode'] = y_decode
            batch['decode_len'] = decode_len
            batch['is_nan_encode'] = is_nan_encode
            batch['is_nan_decode'] = is_nan_decode

            yield batch
