import sys
import random
import json,h5py

import numpy as np
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from data_utils import RehearsalBatchSampler, FixedBufferRehearsalBatchSampler, OriginalDataFixedBufferRehearsalBatchSampler
from torch.utils.data.sampler import SubsetRandomSampler


# 一个问题一个dic
def dictoflists2listofdicts(dictoflists):
    """
    :param dictoflists: {[qid],[q],...}
    :return: [{q1},{q2},...]
    """
    listofdicts = []
    for i in range(len(dictoflists['qid'])):
        tmp = {}
        for k in dictoflists:
            tmp[k] = dictoflists[k][i]
        listofdicts.append(tmp)
    return listofdicts


# 针对特征数据，返回按问题类型排序的列表或打乱（伪随机）的列表
def format_feats_data(h5file, config, num_classes, arrangement = 'random', data_subset = 1.0):
    mem_feat = dict()
    for dset in h5file.keys():
        if config.use_lstm and dset == 'qfeat':
            mem_feat[dset] = [0]*len(h5file['qid'])
        else:
            mem_feat = h5file[dset][:]
    mem_feat = dictoflists2listofdicts(mem_feat)
    mem_feat = mem_feat[:int(len(mem_feat)*data_subset)]

    # 答案空间有限
    data = []
    for d in mem_feat:
        if d['aidx'] < num_classes:
            data.append(d)

    # iid    {'train': 'random', 'val': 'random'}
    # else   {'train': 'qtypeidx', 'val': 'qtypeidx'}
    if arrangement != 'random':
        data = sorted(data, key=lambda k: k[arrangement])
    elif arrangement == 'random':
        random.Random(666).shuffle(data)
    return data


def build_target(ten_aidx, config):
    scores = torch.zeros((config.num_classes))
    ans_freq = Counter(ten_aidx)
    for c in ans_freq:
        if c<config.num_classes and c!= -1:
            scores[c] = min(ans_freq[c]*0.3, 1)
    return scores


class VQADataset(Dataset):
    def __init__(self, data, config, split, **kwargs):
        if split == 'train' and config.arrangement[split]!='random':
            arr = config.arrangement[split]
            all_keys = list(set([d[arr] for d in data]))
            random.Random(666).shuffle(all_keys)
            keymap = {idx: key for idx,key in enumerate(all_keys)}
            data = sorted(data,key = lambda k: keymap[k[arr]])
            for idx, _ in enumerate(data):
                if keymap[data[idx][config.arrangement[split]]]<config.only_first_k[split]:
                    continue
                else:
                    break
        elif config.arrangement[split] != 'random':
            for idx, _ in enumerate(data):
                if data[idx][config.arrangement[split]]<config.only_first_k[split]:
                    continue
                else:
                    break
        else:
            idx = len(data)-1

        data = data[:idx+1]
        self.data = data
        self.split = split
        self.map = json.load(open(config.map_path))
        self.config = config
        self.d = config.d

    def __len__(self):
        if self.config.fetch_all:
            return len(set([dp[self.config.arrangement[self.split]] for dp in self.data]))
        else:
            return len(self.data)

    def __getitem__(self, index):
        if not hasattr(self,'image'):
            # 在这个位置读取h5py
            self.image = h5py.File(self.config.feat_path, 'r')

        if self.config.fetch_all:
            all_valid = [i for i, dp in enumerate(self.data) if dp[self.config.arrangement[self.split]]==index]
            data_batch = (self.get_datapoint(i) for i in all_valid)
            return data_batch
        else:
            return self.get_datapoint(index)

    def get_datapoint(self, index):
        dp = self.data[index]
        iid = str(dp['iid'])
        image_index = self.map['image_id_to_ix'][str(iid)]

        # false
        if self.config.use_pooled:
            image = self.image[image_index]
            image = np.mean(image, axis=0)
        else:
            image = self.image[image_index]

        l=30

        qseq = torch.ones(l).long()*self.d.ntoken
        qtokens = self.d.tokenize(dp['q'],False)
        qlen = len(qtokens)
        qseq[:qlen] = torch.from_numpy(np.array(qtokens[:l-1])).long()

        aidx = dp['aidx']

        return qseq, image, dp['qid'], dp['iid'], aidx, dp['ten_aidx'], qlen


class VQAFeatsDataset(Dataset):
    def __init__(self, data, config, split, mem_feat, **kwargs):
        if split == 'train' and config.arrangement[split]!='random':
            arr = config.arrangement[split]
            all_keys = list(set([d[arr] for d in data]))
            random.Random(666).shuffle(all_keys)
            keymap = {idx: key for idx,key in enumerate(all_keys)}
            data = sorted(data, key = lambda k: keymap[k[arr]])

            for idx, _ in enumerate(data):
                if keymap[data[idx][config.arrangement[split]]]<config.only_first_k[split]:
                    continue
                else:
                    break
        elif config.arrangement[split] != 'random':
            for idx, _ in enumerate(data):
                if data[idx][config.arrangement[split]]<config.only_first_k[split]:
                    continue
                else:
                    break
        else:
            idx = len(data)-1

        data = data[:idx+1]

        self.data = data
        self.split = split
        self.map = json.load(open(config.map_path))
        self.config = config
        self.d = config.d
        if config.load_in_memory:
            self.feat = mem_feat

    def __len__(self):
        if self.config.fetch_all:
            return len(set([dp[self.config.arrangement[self.split]] for dp in self.data]))
        else:
            return len(self.data)

    def __getitem__(self, index):
        if not hasattr(self,'feat'):
            # 在这个位置读取h5py
            self.feat = h5py.File(self.config.feat_path, 'r')

        if self.config.fetch_all:
            all_valid = [i for i, dp in enumerate(self.data) if dp[self.config.arrangement[self.split]]==index]
            data_batch = (self.get_datapoint(i) for i in all_valid)
            return data_batch
        else:
            return self.get_datapoint(index)

    def get_datapoint(self, index):
        dp = self.data[index]
        iid = str(dp['iid'])
        feat_index = self.map['image_id_to_ix'][str(iid)]

        # false
        if self.config.use_pooled:
            imfeat = self.feat['image_features'][feat_index]
            imfeat = np.mean(imfeat, axis=0)
        else:
            imfeat = self.feat['image_features'][feat_index]

        qfeat = dp['qfeat']
        imfeat = imfeat.astype('float32') / (np.linalg.norm(imfeat, axis=1, keepdims=True) + 1e-8)

        if self.config.mkii:
            codebook_index = self.feat['codebook_indices'][feat_index]
        l=30

        qseq = torch.ones(l).long()*self.d.ntoken
        qtokens = self.d.tokenize(dp['q'],False)
        qlen = len(qtokens)
        qseq[:qlen] = torch.from_numpy(np.array(qtokens[:l-1])).long()

        if self.config.soft_targets:
            aidx = build_target(dp['ten_aidx'], self.config)
        else:
            aidx = dp['aidx']

        if self.config.mkii:
            return qfeat, qseq, imfeat, codebook_index, dp['qid'], aidx, dp['ten_aidx'], qlen
        else:
            return qfeat, qseq, imfeat, dp['qid'], dp['iid'], aidx, dp['ten_aidx'], qlen


def collate_batch(data_batch):
    data_batch.sort(key=lambda x: x[-1], reverse=True)
    return torch.utils.data.dataloader.default_collate(data_batch)


def build_dataloaders(config, data_type, mem_feat, **kwargs):
    print('Loading Train Data !')
    train_h5file = h5py.File(config.train_file, 'r')
    print('Filtering Train Data !')
    # full
    if config.train_on == 'valid':
        nc = config.num_classes
    else:
        nc = sys.maxsize

    train_data = format_feats_data(train_h5file, config, num_classes = nc, arrangement = config.arrangement['train'], data_subset = config.data_subset)

    if data_type == 0:
        train_dataset = VQADataset(train_data, config, 'train', mem_feat)
    else:
        train_dataset = VQAFeatsDataset(train_data, config, 'train', mem_feat)


    print('Loading Test Data')
    val_h5file = h5py.File(config.val_file, 'r')
    print('Filtering Test Data')
    # full
    if config.test_on == 'valid':
        nc = config.num_classes
    else:
        nc = sys.maxsize

    val_data = format_feats_data(val_h5file, config, num_classes = nc, arrangement = config.arrangement['train'], data_subset = config.data_subset)

    if data_type == 0:
        val_dataset = VQADataset(val_data, config, 'val', mem_feat)
    else:
        val_dataset = VQAFeatsDataset(val_data, config, 'val', mem_feat)


    if config.fetch_all:
        train_dataloader = train_dataset
        val_dataloader = val_dataset
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False, num_workers=12, collate_fn=collate_batch)
        val_dataloader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False, num_workers=8,collate_fn=collate_batch)

    return train_dataloader, val_dataloader


def build_base_init_dataloader(dataset, data_indices, batch_size):
    index_sampler = SubsetRandomSampler(data_indices)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=index_sampler, num_workers=8, collate_fn=collate_batch)
    return loader


def build_rehearsal_dataloader(dataset, rehearsal_ixs, num_rehearsal_samples):
    rehearsal_batch_sampler = RehearsalBatchSampler(rehearsal_ixs, num_rehearsal_samples)
    loader = DataLoader(dataset, batch_sampler=rehearsal_batch_sampler)
    return loader


def build_original_dataloader_with_limited_buffer(dataset, rehearsal_ixs, num_rehearsal_samples, max_buffer_size, buffer_replacement_strategy):
    rehearsal_batch_sampler = OriginalDataFixedBufferRehearsalBatchSampler(max_buffer_size, num_rehearsal_samples, buffer_replacement_strategy)
    loader = DataLoader(dataset, batch_sampler=rehearsal_batch_sampler)
    return loader


def build_rehearsal_dataloader_with_limited_buffer(dataset, rehearsal_ixs, num_rehearsal_samples, max_buffer_size, buffer_replacement_strategy):
    rehearsal_batch_sampler = FixedBufferRehearsalBatchSampler(max_buffer_size, num_rehearsal_samples, buffer_replacement_strategy)
    loader = DataLoader(dataset, batch_sampler=rehearsal_batch_sampler)
    return loader


def main():
    pass


if __name__ == '__main__':
    main()