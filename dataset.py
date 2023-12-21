import csv
from datetime import datetime
import numpy as np
from utils import latlon2quadkey
from torch.utils.data import Dataset
from collections import defaultdict
import copy
import math


class LBSNDataset(Dataset):
    def __init__(self, filename, k_d = 1.0):
        self.loc2idx = {'<pad>': 0}
        self.loc2gps = {'<pad>': (0.0, 0.0)}
        self.idx2loc = {0: '<pad>'}
        # (latitude, longitude) tuple
        self.idx2gps = {0: (0.0, 0.0)}
        self.loc2count = {}
        self.n_loc = 1
        self.build_vocab(filename)
        print(f'{self.n_loc} locations')
        # user_seq_array,user2idx,n_users,session_dic
        self.user_seq,self.user2idx, self.n_user,self.session_dic= self.processing(filename,k_d)
        print(f'{len(self.user_seq)} users')

    def region_stats(self):
        num_reg_locs = []
        for reg in self.region2loc:
            num_reg_locs.append(len(self.region2loc[reg]))
        num_reg_locs = np.array(num_reg_locs, dtype=np.int32)
        print("min #loc/region: {:d}, with {:d} regions".format(np.min(num_reg_locs), np.count_nonzero(num_reg_locs == 1)))
        print("max #loc/region:", np.max(num_reg_locs))
        print("avg #loc/region: {:.4f}".format(np.mean(num_reg_locs)))
        hist, bin_edges = np.histogram(num_reg_locs, bins=[1, 3, 5, 10, 20, 50, 100, 200, np.max(num_reg_locs)])
        for i in range(len(bin_edges) - 1):
            print("#loc in [{}, {}]: {:d} regions".format(math.ceil(bin_edges[i]), math.ceil(bin_edges[i + 1] - 1), hist[i]))

    def build_vocab(self, filename, min_freq=0):
        for line in open(filename):
            line = line.strip().split(',')
            loc = line[1]
            if loc == 'loc':
                continue
            coordinate = line[3], line[4]
            self.add_location(loc, coordinate)
        if min_freq > 0:
            self.n_loc = 1
            self.loc2idx = {'<pad>': 0}
            self.idx2loc = {0: '<pad>'}
            self.idx2gps = {0: (0.0, 0.0)}
            for loc in self.loc2count:
                if self.loc2count[loc] >= min_freq:
                    self.add_location(loc, self.loc2gps[loc])
        self.locidx2freq = np.zeros(self.n_loc - 1, dtype=np.int32)
        for idx, loc in self.idx2loc.items():
            if idx != 0:
                self.locidx2freq[idx - 1] = self.loc2count[loc]

    def add_location(self, loc, coordinate):
        if loc not in self.loc2idx:
            self.loc2idx[loc] = self.n_loc
            self.loc2gps[loc] = coordinate
            self.idx2loc[self.n_loc] = loc
            self.idx2gps[self.n_loc] = coordinate
            if loc not in self.loc2count:
                self.loc2count[loc] = 1
            self.n_loc += 1
        else:
            self.loc2count[loc] += 1

    def processing(self, filename, min_freq=20):
        user_seq = {}
        user_seq_array = list()
        region2idx = {}
        idx2region = {}
        regidx2loc = defaultdict(set)
        n_region = 1
        user2idx = {}
        n_users = 1
        for line in open(filename):
            user, time, lat, lon, loc = line.strip().split('\t')
            if loc not in self.loc2idx:
                continue
            time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')
            time_idx = time.weekday() * 24 + time.hour + 1
            loc_idx = self.loc2idx[loc]
            region = latlon2quadkey(float(lat), float(lon), 11)
            if region not in region2idx:
                region2idx[region] = n_region
                idx2region[n_region] = region
                n_region += 1
            region_idx = region2idx[region]
            regidx2loc[region_idx].add(loc_idx)
            if user not in user_seq:
                user_seq[user] = list()
            user_seq[user].append([loc_idx, time_idx, region_idx, time])
        for user, seq in user_seq.items():
            if len(seq) >= min_freq:
                user2idx[user] = n_users
                user_idx = n_users
                seq_new = list()
                tmp_set = set()
                cnt = 0
                for loc, t, region, _ in sorted(seq, key=lambda e: e[3]):
                    if loc in tmp_set:
                        seq_new.append((user_idx, loc, t, region, True))
                    else:
                        seq_new.append((user_idx, loc, t, region, False))
                        tmp_set.add(loc)
                        cnt += 1
                if cnt > min_freq / 2:
                    n_users += 1
                    user_seq_array.append(seq_new)
        return user_seq_array, user2idx, region2idx, n_users, n_region, regidx2loc, 169

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        return self.user_seq[idx]

    def split(self, max_len=128):
        train_ = copy.copy(self)
        test_ = copy.copy(self)
        valid_ = copy.copy(self)
        train_seq = list()
        test_seq = list()
        valid_seq = list()
        for u in range(len(self)):
            seq = self[u]
            traj_id = seq[0][9] # first traj_id
            traj = []
            seq_traj = []
            for iter in seq:
                if iter[9] != traj_id:
                    seq_traj.append(traj)
                    traj = []
                    traj_id = iter[9]
                traj.append(iter)
            seq_traj.append(traj)
            seq_use = []
            for t in seq_traj:
                seq_use += t
                if len(seq_use) < 5:
                    continue
                if len(seq_use) > max_len:
                    seq_use = seq_use[-max_len:]
                divide = t[-1][-1] # traj last pls is train or test or valid
                src = seq_use[:-1]
                trg = seq_use[1:]
                if divide == 'train':
                    train_seq.append((src,trg))
                elif divide == 'validation':
                    valid_seq.append((src,[trg[-1]]))
                elif divide == 'test':
                    test_seq.append((src,[trg[-1]]))
        train_.user_seq = train_seq
        test_.user_seq = sorted(test_seq, key=lambda e: len(e[0]))
        valid_.user_seq = sorted(valid_seq, key=lambda e: len(e[0]))
        return train_, test_, valid_

    # def split(self, max_len=128):
    #     train_ = copy.copy(self)
    #     test_ = copy.copy(self)
    #     valid_ = copy.copy(self)
    #     train_seq = list()
    #     test_seq = list()
    #     valid_seq = list()
    #     for u in range(len(self)):
    #         seq = self[u]
    #         i = len(seq)-1
    #         for b in range(math.floor((i + max_len - 1) // max_len)):
    #             if (i - b * max_len) > max_len*1.1:
    #                 trg = seq[(i - (b + 1) * max_len): (i - b * max_len)]
    #                 src = seq[(i - (b + 1) * max_len - 1): (i - b * max_len - 1)]
    #                 train_seq.append((src, trg))
    #             else:
    #                 trg = seq[1: (i - b * max_len)]
    #                 src = seq[0: (i - b * max_len - 1)]
    #                 train_seq.append((src, trg))
    #                 break
    #             if len(trg)>128 or len(src)>128:
    #                 print('false')
    #                 break
    #         # for b in range(i-3-max_len):
    #         #     src = seq[b:b+max_len]
    #         #     trg = seq[b+1:b+max_len+1]
    #         #     train_seq.append((src,trg))
            
    #         valid_seq.append((seq[max(0, -max_len+i-1):i-1], seq[i-1:i]))
    #         test_seq.append((seq[max(0, -max_len+i):i], seq[i:i+1]))
    #     train_.user_seq = train_seq
    #     test_.user_seq = sorted(test_seq, key=lambda e: len(e[0]))
    #     valid_.user_seq = sorted(valid_seq, key=lambda e: len(e[0]))
    #     return train_, test_, valid_


class NegInclLSBNDataset(Dataset):
    def __init__(self, test_dataset, eval_sort_samples):
        self.user_seq = test_dataset.user_seq
        self.sort_samples = eval_sort_samples

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        return self.user_seq[idx][0], self.user_seq[idx][1], self.sort_samples[idx]