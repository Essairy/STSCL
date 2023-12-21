import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
import random
    
import math
def haversine_distance(lat1, lon1, lat2, lon2):
    # 将度数转换为弧度
    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))
    
    # 地球半径（千米）
    R = 6371.0
    
    # 使用球面三角法计算距离
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance

def collect_fn_step(batch,sampler,k,train=True):
    src,trg = zip(*batch)
    user, loc, time, wday, hour, session, lon, lat, cat = [], [], [], [], [], [], [], [], [] 
    data_size = []
    trg_ = []
    pos_ = []
    pos_category_,pos_hour_ = [],[]
    # user_idx, loc, time, tw_day, tw_wday, tw_hour, lon, lat, session, trajectory_id, cat, SplitTag
        
    for e in src:
        u_, l_, t_, d_, wd_, h_, lo, la, s, traj, c, split_tag = zip(*e)
        data_size.append(len(u_))
        user.append(torch.tensor(u_))
        loc.append(torch.tensor(l_))
        time.append(torch.tensor(t_))
        wday.append(torch.tensor(wd_))
        hour.append(torch.tensor(h_))
        lon_f = [float(val) for val in lo]
        lat_F = [float(val) for val in la]
        lon.append(torch.tensor(lon_f))
        lat.append(torch.tensor(lat_F))
        cat_F = [float(category) for category in c]
        cat.append(torch.tensor(cat_F))
        session.append(torch.tensor(s))
    for i, seq in enumerate(trg):
        pos = torch.tensor([[e[1]] for e in seq])
        pos_.append(pos)
        pos_category =torch.tensor([[float(e[10])] for e in seq])
        pos_hour = torch.tensor([[e[5]] for e in seq])
        pos_hour_.append(pos_hour)
        pos_category_.append(pos_category)
        neg, probs = sampler(seq, k, user=seq[0][0])
        trg_.append(torch.cat([pos, neg], dim=-1))

    pos_ = pad_sequence(pos_,batch_first=True)
    trg_ = pad_sequence(trg_,batch_first=True) # bs*trg_len*neg
    trg_ = trg_.permute(2, 1, 0).contiguous().view(-1, trg_.size(0))    # (trg_len*neg)*bs
    user_ = pad_sequence(user,batch_first=True)
    loc_ = pad_sequence(loc,batch_first=True)
    wday_ = pad_sequence(wday,batch_first=True)
    hour_ = pad_sequence(hour,batch_first=True)
    t_  = pad_sequence(time,batch_first=True)
    lon_  = pad_sequence(lon,batch_first=True)
    lat_  = pad_sequence(lat,batch_first=True)
    session = pad_sequence(session,batch_first=True)
    cat_ = pad_sequence(cat,batch_first=True)
    pos_category_ = pad_sequence(pos_category_,batch_first=True).squeeze(-1)
    pos_hour_ = pad_sequence(pos_hour_,batch_first=True).squeeze(-1)

    return user_,loc_,wday_,hour_,t_,lon_,lat_,session,pos_category_,pos_hour_,pos_,trg_,data_size

            

