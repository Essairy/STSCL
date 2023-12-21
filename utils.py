from collections import OrderedDict
import os
import json
import torch
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import random
import math

import pickle as _pickle


EarthRadius = 6378137
MinLatitude = -85.05112878
MaxLatitude = 85.05112878
MinLongitude = -180
MaxLongitude = 180


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    try:
        model.load_state_dict(checkpoint['model_state_dict'], False)
        # model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            k_ = k[7:]
            # remove 'module.' of DistributedDataParallel instance
            state_dict[k_] = v
        model.load_state_dict(state_dict)

    model.eval()
    return model

def save_model(model, model_dir, current_epoch, last_best_epoch=None, pretrain = False):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if last_best_epoch is not None and current_epoch != last_best_epoch:
        # model_state_file = os.path.join(model_dir, 'model_{}_best.pth'.format(current_epoch))
        model_state_file = os.path.join(model_dir, 'model_best.pth')
    else:
        model_state_file = os.path.join(model_dir, 'model_{}.pth'.format(current_epoch))
    if pretrain:
        model_state_file = os.path.join(model_dir, 'model_pretrain{}.pth'.format(current_epoch))
        print('save at {}'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

def generate_square_mask(sz, device):
    mask = (torch.triu(torch.ones(sz, sz).to(device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_decoder_mask(sz, ds, device, last_n=5, test=False):
    mask = (torch.triu(torch.ones(sz, sz).to(device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask_front = torch.tril(torch.ones(sz, sz).to(device) == 1, diagonal=-last_n)
    mask_front = mask_front.float().masked_fill(mask_front == 1, float('-inf')).masked_fill(mask_front == 0, float(0.0))
    mask = mask.unsqueeze(0).repeat(len(ds), 1, 1)
    for idx in range(len(ds)):
        if not test:
            mask[idx, :ds[idx], :] += mask_front[:ds[idx], :]
        else:
            mask[idx, 0, :] = mask[idx, ds[idx] - 1, :] + mask_front[ds[idx] - 1, :]
    if test:
        mask = mask[:, 0, :].unsqueeze(1)
    return mask

def generate_test_mask(trg_len, src_len, device):
    assert src_len > trg_len
    m1 = generate_square_mask(trg_len, device)
    m2 = torch.zeros(trg_len, src_len - trg_len).to(device)
    return torch.cat([m2, m1], 1)

def serialize(obj, path, in_json=False):
    if in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            _pickle.dump(obj, file)


def unserialize(path):
    suffix = os.path.basename(path).split(".")[-1]
    if suffix == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:
            return _pickle.load(file)


def collect_fn(batch, data_source, sampler, k=5):
    src, trg = zip(*batch)
    user, loc, time, region = [], [], [], []
    data_size = []
    trg_ = []
    trg_probs_= []
    for e in src:
        u_, l_, t_, r_, b_ = zip(*e)
        data_size.append(len(u_))
        user.append(torch.tensor(u_))
        loc.append(torch.tensor(l_))
        time.append(torch.tensor(t_))
        region.append(torch.tensor(r_))
    user_ = pad_sequence(user, batch_first=True)
    loc_ = pad_sequence(loc, batch_first=True)
    time_ = pad_sequence(time, batch_first=True)
    region_ = pad_sequence(region, batch_first=True)
    for i, seq in enumerate(trg):
        pos = torch.tensor([[e[1]] for e in seq])
        neg, probs = sampler(seq, k, user=seq[0][0])
        trg_.append(torch.cat([pos, neg], dim=-1))
        trg_probs_.append(probs)
    trg_ = pad_sequence(trg_, batch_first=True)
    trg_probs_ = pad_sequence(trg_probs_, batch_first=True, padding_value=1.0)
    trg_ = trg_.permute(2, 1, 0).contiguous().view(-1, trg_.size(0))
    trg_nov_ = [[not e[-1] for e in seq] for seq in trg]    
    return user_.t(), loc_.t(), time_.t(), region_.t(), trg_, trg_nov_, trg_probs_, data_size

def collect_fn_neg_included(batch, data_source):
    src, trg, neg_samples = zip(*batch)
    user, loc, time, region = [], [], [], []
    data_size = []
    trg_ = []
    trg_probs_= []
    for e in src:
        u_, l_, t_, r_, b_ = zip(*e)
        data_size.append(len(u_))
        user.append(torch.tensor(u_))
        loc.append(torch.tensor(l_))
        time.append(torch.tensor(t_))
        region.append(torch.tensor(r_))
    user_ = pad_sequence(user, batch_first=True)
    loc_ = pad_sequence(loc, batch_first=True)
    time_ = pad_sequence(time, batch_first=True)
    region_ = pad_sequence(region, batch_first=True)
    for seq, neg in zip(trg, neg_samples):
        pos = torch.tensor([[e[1]] for e in seq])
        neg = torch.tensor(np.expand_dims(neg, axis=0), dtype=torch.long)
        trg_.append(torch.cat([pos, neg], dim=-1))
    trg_ = pad_sequence(trg_, batch_first=True)
    trg_ = trg_.permute(2, 1, 0).contiguous().view(-1, trg_.size(0))
    return user_.t(), loc_.t(), time_.t(), region_.t(), trg_, None, None, data_size

def collect_fn_export(batch, data_source, sampler, k=5):
    src, trg = zip(*batch)
    user, loc, time, region = [], [], [], []
    data_size = []
    trg_ = []
    for e in src:
        u_, l_, t_, r_, b_ = zip(*e)
        data_size.append(len(u_))
        user.append(np.array(u_))
        loc.append(np.array(l_))
        time.append(np.array(t_))
        region.append(np.array(r_))
    for seq in trg:
        pos = np.array([[e[1]] for e in seq])
        neg, _ = sampler(seq, k)
        neg = neg.numpy()
        trg_.append(np.concatenate([pos, neg], axis=-1))
    return user, loc, time, region, trg_, data_size

def get_visited_locs(dataset):
    user_visited_locs = {}
    for u in range(len(dataset.user_seq)):
        seq = dataset.user_seq[u]
        user = seq[0][0]
        user_visited_locs[user] = set()
        for i in reversed(range(len(seq))):
            if not seq[i][-1]:
                break
        user_visited_locs[user].add(seq[i][1])
        seq = seq[:i]
        for check_in in seq:
            user_visited_locs[user].add(check_in[1])
    return user_visited_locs

class LadderSampler(Sampler):
    def __init__(self, data_source, batch_sz, fix_order=False):
        super(LadderSampler, self).__init__(data_source)
        self.data = [len(e[0]) for e in data_source]
        self.batch_size = batch_sz * 100
        self.fix_order = fix_order

    def __iter__(self):
        if self.fix_order:
            d = zip(self.data, np.arange(len(self.data)), np.arange(len(self.data)))
        else:
            d = zip(self.data, np.random.permutation(len(self.data)), np.arange(len(self.data)))
        d = sorted(d, key=lambda e: (e[1] // self.batch_size, e[0]), reverse=True)
        return iter(e[2] for e in d)

    def __len__(self):
        return len(self.data)


def reset_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clip(n, minValue, maxValue):
    return min(max(n, minValue), maxValue)

def map_size(levelOfDetail):
    return 256 << levelOfDetail

def latlon2pxy(latitude, longitude, levelOfDetail):
    latitude = clip(latitude, MinLatitude, MaxLatitude)
    longitude = clip(longitude, MinLongitude, MaxLongitude)

    x = (longitude + 180) / 360
    sinLatitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)

    mapSize = map_size(levelOfDetail)
    pixelX = int(clip(x * mapSize + 0.5, 0, mapSize - 1))
    pixelY = int(clip(y * mapSize + 0.5, 0, mapSize - 1))
    return pixelX, pixelY

def txy2quadkey(tileX, tileY, levelOfDetail):
    quadKey = []
    for i in range(levelOfDetail, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tileX & mask) != 0:
            digit += 1
        if (tileY & mask) != 0:
            digit += 2
        quadKey.append(str(digit))

    return ''.join(quadKey)

def pxy2txy(pixelX, pixelY):
    tileX = pixelX // 256
    tileY = pixelY // 256
    return tileX, tileY

def latlon2quadkey(lat,lon,level):
    pixelX, pixelY = latlon2pxy(lat, lon, level)
    tileX, tileY = pxy2txy(pixelX, pixelY)
    return txy2quadkey(tileX, tileY,level)

def groupby_mean(value: torch.Tensor, labels: torch.LongTensor, device):
    # result_mean, result_label = groupby_mean_nondeterministically(value, labels, device)
    # result_mean, result_label = groupby_mean_slow(value, labels, device)
    # result_mean, result_label = groupby_mean_normal(value, labels, device)
    # result_mean, result_label = groupby_mean_matrix(value, labels, device)
    result_mean, result_label = groupby_mean_sparse(value, labels, device)
    return result_mean, result_label

def groupby_mean_sparse(value: torch.Tensor, labels: torch.LongTensor, device):
    uniques = labels.unique().tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}

    remap_labels = torch.LongTensor(list(map(key_val.get, labels)))

    indices = torch.stack([remap_labels, torch.arange(len(value))], dim=0).to(device)
    values = torch.ones_like(remap_labels, dtype=torch.float32).to(device)
    M = torch.sparse.FloatTensor(indices, values, (len(uniques), len(value)))

    diags = torch.sparse.sum(M, dim=1).to_dense()   # 每一个label出现的次数
    diags = torch.pow(diags, -1)
    diags_lookup = diags[remap_labels]
    norm_M = torch.sparse.FloatTensor(indices, diags_lookup, (len(uniques), len(value)))

    result_mean = torch.sparse.mm(norm_M, value)    # 出现的次数越多，缩小的越多
    result_label = torch.LongTensor(list(map(val_key.get, range(len(uniques))))).to(device)
    origin_label = torch.LongTensor(uniques).to(device)
    return result_mean, result_label

def groupby_last(value: torch.Tensor, labels: torch.LongTensor, device):
    uniques = labels.unique().tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}

    remap_labels = torch.LongTensor(list(map(key_val.get, labels))) # 获取重新编码的id

    indices = torch.stack([remap_labels, torch.arange(len(value))], dim=0).to(device)
    values = torch.ones_like(remap_labels, dtype=torch.float32).to(device)
    M = torch.sparse.FloatTensor(indices, values, (len(uniques), len(value)))
    def get_last(M):
        values = M._values()
        indices = M._indices()

        # 获取每行最后一个元素的位置
        row_ends = torch.zeros(M.size(0), dtype=torch.long)
        for i in range(M.size(0)):
            row = indices[0]  # 获取行索引
            # 找到当前行的所有索引
            row_indices = indices[1][row == i]
            # 获取当前行的最后一个元素的位置
            if len(row_indices) > 0:
                row_ends[i] = row_indices.max()

        # 创建一个新的稀疏张量，每行的所有1中只有最后一个出现的为1，其他为0
        new_values = torch.zeros_like(values)
        for i in range(len(values)):
            row = indices[0][i]  # 获取当前值所在的行
            col = indices[1][i]  # 获取当前值所在的列
            if col == row_ends[row]:
                new_values[i] = 1

        # 创建新的稀疏张量
        new_sparse_tensor = torch.sparse_coo_tensor(indices, new_values, M.size())
        return new_sparse_tensor

    diags = get_last(M).to_dense() 
    result_mean = torch.sparse.mm(diags, value)    # 出现的次数越多，缩小的越多
    result_label = torch.LongTensor(list(map(val_key.get, range(len(uniques))))).to(device)
    origin_label = torch.LongTensor(uniques).to(device)

    return result_mean, result_label
