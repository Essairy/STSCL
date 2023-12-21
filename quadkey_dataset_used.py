from dataset import LBSNDataset
from datetime import datetime
import numpy as np
from utils import latlon2quadkey
from collections import defaultdict
from nltk import ngrams
from torchtext.legacy import data
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter

from utils import serialize, unserialize
import argparse
import os

LOD = 17

class QuadKeyLBSNDataset(LBSNDataset):
    def __init__(self, filename):
        super().__init__(filename)

    def processing(self, filename, min_freq=5):
        user_seq = {}
        user_seq_array = list()
        region2idx = {}
        idx2region = {}
        regidx2loc = defaultdict(set)
        n_region = 1
        user2idx = {}
        n_users = 1
        session_dic = {}
        geo_session_dic = {}

        loc_gps = np.array(list(self.idx2gps.values())[1:])
        k = 9
        # 初始化K-means模型
        kmeans = KMeans(n_clusters=k)
        # 拟合数据
        kmeans.fit(loc_gps)
        # 获取簇中心
        # centroids = kmeans.cluster_centers_
        # 获取每个样本所属的簇
        labels = kmeans.labels_
        self.idx2label = {}
        # 去掉了0
        for i in range(len(loc_gps)):
            self.idx2label[i+1] = labels[i]

        for line in open(filename):
            line = line.strip().split('\t')
            if len(line) < 5:
                print('read line data false')
                continue
            #user, time, lat, lon, loc = line
            user, loc, lon, lat, time = line
            if loc not in self.loc2idx:
                continue
            #time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')
            time = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
            tw_hour = time.hour
            tw_wday = time.weekday()
            loc_idx = self.loc2idx[loc]
            region = latlon2quadkey(float(lat), float(lon), LOD)
            if region not in region2idx:
                region2idx[region] = n_region
                idx2region[n_region] = region
                n_region += 1
            region_idx = region2idx[region]
            regidx2loc[region_idx].add(loc_idx)
            if user not in user_seq:
                user_seq[user] = list()
            time_sec = time.timestamp()
            user_seq[user].append([loc_idx, region_idx, region, lon, lat, time_sec, tw_wday, tw_hour])

        session = -1
        # geo_session = -1
        session2locs = {}
        for user, seq in user_seq.items():
            seq_len = len(seq)
            if seq_len >= min_freq:
                user2idx[user] = n_users
                user_idx = n_users
                seq_new = list()
                tmp_set = set()
                cnt = 0
                session += 1
                # geo_session += 1
                pre_time = 0
                loc_list = []
                loc_id_list = []
                pre_lat,pre_lon = 0,0
                temp = True

                for loc,  region_idx, region_quadkey, lon, lat, time, tw_wday, tw_hour in sorted(seq, key=lambda e: e[5]):
                    cur_time = time
                    temp = True
                    for p_lat,p_lon in loc_list:
                        if haversine_distance(lat,lon,p_lat,p_lon) > 1.5:
                            temp = False # 距离外
                            break
                    time_gap = cur_time-pre_time

                    if pre_time == 0 or len(loc_list) == 0:
                        loc_list.append((pre_lat,pre_lon))  
                        loc_id_list.append(loc)
                    elif time_gap<5400 and temp == True:
                        loc_list.append((pre_lat,pre_lon))  
                        loc_id_list.append(loc)
                    else:
                        session2locs[session] = loc_id_list
                        session += 1
                        loc_list,loc_id_list = [],[]
                        loc_list.append((pre_lat,pre_lon))  
                        loc_id_list.append(loc)
                    pre_lat,pre_lon = lat,lon
                    pre_time = time

                    if loc in tmp_set:
                        seq_new.append((user_idx, loc, time, tw_wday, tw_hour, region_quadkey,session, True))
                    else:
                        seq_new.append((user_idx, loc, time, tw_wday, tw_hour, region_quadkey,session, False))
                        tmp_set.add(loc)
                        cnt += 1
                session2locs[session] = loc_id_list
                if cnt > min_freq:
                    n_users += 1
                    user_seq_array.append(seq_new)
    
        all_quadkeys = []
        for u in range(len(user_seq_array)):
            seq = user_seq_array[u]
            for i in range(len(seq)):
                check_in = seq[i]
                region_quadkey = check_in[5]
                region_quadkey_bigram = ' '.join([''.join(x) for x in ngrams(region_quadkey, 6)])
                region_quadkey_bigram = region_quadkey_bigram.split()
                # all_quadkeys.append(region_quadkey_bigram)    
                cur_session = check_in[6]
                day = check_in[3]
                hour = check_in[4]

                # get session region
                loc_id_list = session2locs[cur_session]
                region_list = [self.idx2label[i] for i in loc_id_list]
                loc_freq = sorted(Counter(region_list).items(), key=lambda x: x[1], reverse=True)
                label = loc_freq[0][0] # get most appear region

                session_dic[cur_session] = (u,day,hour,label)     # user,loc,time,day,hour,region,session,geo_session,first

                # cur_g_session = check_in[7]
                # geo_session_dic[cur_g_session] = (u,day,hour)

                user_seq_array[u][i] = (check_in[0], check_in[1],check_in[2], check_in[3], check_in[4],region_quadkey_bigram,check_in[6],check_in[7])

        self.loc2quadkey = ['NULL']
        for l in range(1, self.n_loc):
            lat, lon = self.idx2gps[l]
            quadkey = latlon2quadkey(float(lat), float(lon), LOD)
            quadkey_bigram = ' '.join([''.join(x) for x in ngrams(quadkey, 6)])
            quadkey_bigram = quadkey_bigram.split()
            self.loc2quadkey.append(quadkey_bigram)
            all_quadkeys.append(quadkey_bigram)
        
        self.QUADKEY = data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            unk_token=None,
            preprocessing=str.split
        )
        self.QUADKEY.build_vocab(all_quadkeys)

        return user_seq_array,session_dic,geo_session_dic,user2idx, region2idx, n_users, n_region, regidx2loc, 169

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    filename_raw = os.path.join(args.dataset, "totalCheckins.txt")
    filename_clean = os.path.join(args.dataset, "QuadKeyLSBNDataset.data")

    if not os.path.isfile(filename_clean):
        dataset = QuadKeyLBSNDataset(filename_raw)
        serialize(dataset, filename_clean)
    else:
        dataset = unserialize(filename_clean)
    
    count = 0
    length = []
    for seq in dataset.user_seq:
        count += len(seq)
        length.append(len(seq))
    print("#check-ins:", count)
    print("#users:", dataset.n_user - 1)
    print("#locations:", dataset.n_loc - 1)
    print("#median seq len:", np.median(np.array(length)))