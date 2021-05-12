import os

import numpy as np
import torch

def randint(max_val, num_samples):
    rand_vals = {}
    _num_samples = min(max_val, num_samples)
    while True:
        _rand_vals = np.random.randint(0, max_val, num_samples)
        for r in _rand_vals:
            rand_vals[r] = r
            if len(rand_vals) >= _num_samples:
                break

        if len(rand_vals) >= _num_samples:
            break
    return rand_vals.keys()


class MySampler():
    def __init__(self,dataset, max_buffer_size, num_rehearsal_samples, buffer_replacement_strategy, sampling_method):
        self.dataset = dataset
        self.max_buffer_size = max_buffer_size
        self.sampling_method = sampling_method
        self.num_rehearsal_samples = num_rehearsal_samples
        self.buffer_replacement_strategy = buffer_replacement_strategy

        # TODO: qtype 000000 class
        self.per_class_rehearsal_ixs = {} # "class"：[ixs]
        self.class_lens = {} # "class": len(ixs)
        self.total_len = 0
        self.device = 'cuda'
        np.random.seed(os.getpid())
        self.cur_ix = 0

    def find_class_having_max_samples(self):
        max_class = None
        max_num = 0
        for c in self.class_lens:
            class_len = self.class_lens[c]
            if class_len > max_num:
                max_num = class_len
                max_class = c
        return max_class, max_num

    def delete_sample_from_largest_class(self):
        max_class, max_num = self.find_class_having_max_samples()
        if self.buffer_replacement_strategy == 'random':
            del_ix = int(list(randint(max_num, 1))[0])
            del self.per_class_rehearsal_ixs[max_class][del_ix]

        self.class_lens[max_class] -= 1
        self.total_len -= 1

    def update_buffer(self, new_ix, class_id):
        new_ix = int(new_ix)
        class_id = int(class_id)
        if self.total_len >= self.buffer_size:
            self.delete_sample_from_largest_class()
        if class_id not in self.per_class_rehearsal_ixs:
            self.per_class_rehearsal_ixs[class_id] = []
            self.class_lens[class_id] = 0
        self.class_lens[class_id] += 1
        self.per_class_rehearsal_ixs[class_id].append(new_ix)

        self.cur_ix = new_ix
        self.total_len += 1

    def get_rehearsal_item_ix(self, ix):
        """
        Given a random integer 'ix', this function figures out the class and the index
        within the class this refers to, and returns that element.
        :param ix:
        :return:
        """
        cum_sum = 0
        for class_id, class_len in zip(list(self.class_lens.keys()), list(self.class_lens.values())):
            cum_sum += class_len
            if ix < cum_sum:
                class_item_ix = class_len - (cum_sum - ix)
                # print(
                #     f"class_item_ix {class_item_ix} class_len {class_len} len {len(self.per_class_rehearsal_ixs[class_id])}")
                return self.per_class_rehearsal_ixs[class_id][class_item_ix]

    def calc_desision_boundary(self,param):
        w1,b1,w2,b2 = param
        cur = self.dataset[self.cur_ix]
        print(type(w1))
        print(w1.shape)
        # ( w1 * cur + b1 ) * w2 + b2
        return 0

    def get_rehearsal_ix_(self, db):
        ixs = randint(self.total_len, self.num_rehearsal_samples * 10)
        ixs = [self.get_rehearsal_ixs(ix) for ix in ixs]
        # rT * db
        # sort(近的
        return 0

    def __iter__(self, param=None):
        while True:
            if self.sampling_method == "PDS":
                db = self.calc_desision_boundary(param)
                ixs = self.get_rehearsal_ix_(db)
                yield ixs



    def __len__(self):
        return 2 ** 64  # Returning a very large number because we do not want it to stop replaying.
        # The stop criteria must be defined in some other manner.

    def get_state(self):
        return {
            'buffer_size': self.buffer_size,
            'per_class_rehearsal_ixs': self.per_class_rehearsal_ixs,
            'num_rehearsal_samples': self.num_rehearsal_samples,
            'class_lens': self.class_lens,
            'total_len': self.total_len
        }

    def load_state(self, state):
        self.buffer_size = state['buffer_size']
        self.num_rehearsal_samples = state['num_rehearsal_samples']
        self.total_len = state['total_len']
        for c in state['class_lens']:
            self.class_lens[c] = state['class_lens'][c]
            print(f"class len {c}: {self.class_lens[c]}")
        for c in state['per_class_rehearsal_ixs']:
            if c in self.per_class_rehearsal_ixs:
                while len(self.per_class_rehearsal_ixs[c]) > 0:
                    self.per_class_rehearsal_ixs[c].pop()
            else:
                self.per_class_rehearsal_ixs[c] = []
            self.per_class_rehearsal_ixs[c].extend(state['per_class_rehearsal_ixs'][c])

    def get_rehearsal_ixs(self):
        rehearsal_ixs = []
        for c in self.per_class_rehearsal_ixs:
            rehearsal_ixs += self.per_class_rehearsal_ixs[c]
        return rehearsal_ixs

    def get_len_of_rehearsal_ixs(self):
        return len(self.get_rehearsal_ixs())



class MyDataLoader():
    def __init__(self, dataset, max_buffer_size, num_rehearsal_samples, buffer_replacement_strategy):
        self.dataset = dataset
        self.max_buffer_size = max_buffer_size
        self.num_rehearsal_samples = num_rehearsal_samples
        self.buffer_replacement_strategy = buffer_replacement_strategy

        self.per_class_rehearsal_ixs = {}  # "class"：[ixs]
        self.per_class_mean_of_Features = {}
        self.class_lens = {}  # "class": len(ixs)
        self.total_len = 0
        self.device = 'cuda'
        self.cur_ix = 0

    def find_class_having_max_samples(self):
        max_class = None
        max_num = 0
        for c in self.class_lens:
            class_len = self.class_lens[c]
            if class_len > max_num:
                max_num = class_len
                max_class = c
        return max_class, max_num

    def delete_sample_from_largest_class(self):
        max_class, max_num = self.find_class_having_max_samples()
        if self.buffer_replacement_strategy == 'random':
            del_ix = int(list(randint(max_num, 1))[0])
            del self.per_class_rehearsal_ixs[max_class][del_ix]

        self.class_lens[max_class] -= 1
        self.total_len -= 1

    def update_buffer(self, new_ix, class_id):
        new_ix = int(new_ix)
        class_id = int(class_id)
        if self.total_len >= self.buffer_size:
            self.delete_sample_from_largest_class()
        if class_id not in self.per_class_rehearsal_ixs:
            self.per_class_rehearsal_ixs[class_id] = []
            self.class_lens[class_id] = 0
        self.class_lens[class_id] += 1
        self.per_class_rehearsal_ixs[class_id].append(new_ix)

        self.cur_ix = new_ix
        self.total_len += 1

    def Mean_of_Features(self):
        keys = list(self.class_lens.keys())
        Mf = []
        for key in keys:
            mf = torch.mean([self.dataset[self.per_class_rehearsal_ixs[key]] for key in keys])
            Mf.append({key: mf})

        return Mf

    def __iter__(self, param=None):
        while True:
            return self.Mean_of_Features()
