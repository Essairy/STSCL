import random
from dataset import LBSNDataset
from datetime import datetime

from datautil import haversine_distance

from utils import serialize, unserialize
import argparse
import os


class stscl_dataset(LBSNDataset):
    def __init__(self, filename, k_d):
        super().__init__(filename, k_d)

    def processing(self, filename, k_d, min_freq=0):
        user_seq = {}
        user_seq_array = list()
        user2idx = {}
        n_users = 1
        session_dic = {}
        print('k_d=',k_d)

        for line in open(filename):
            line = line.strip().split(',')
            user, loc, time, lat, lon, trajectory_id,cat,SplitTag  = line
            if loc == 'loc':    # skip title
                continue
            if loc not in self.loc2idx:
                continue
            time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
            tw_hour = time.hour
            tw_wday = time.weekday()
            tw_day = time.day
            loc_idx = self.loc2idx[loc]
            if user not in user_seq:
                user_seq[user] = list()
            time_sec = time.timestamp()
            user_seq[user].append([loc_idx, lon, lat, time_sec, tw_day, tw_wday, tw_hour,trajectory_id,cat,SplitTag])

        session = -1
        for user, seq in user_seq.items():
            seq_len = len(seq)
            if seq_len >= 3:
            # if True:
                user2idx[user] = n_users
                user_idx = n_users
                seq_new = list()
                cnt = 0
                
                session += 1
                pre_time = 0
                loc_list = []
                loc_id_list = []
                pre_lat, pre_lon = 0,0
                temp = True

                for loc_idx, lon, lat, time_sec, tw_day, tw_wday, tw_hour, trajectory_id, cat, SplitTag in sorted(seq, key=lambda e: e[3]):
                    lat = float(lat)
                    lon = float(lon)
                    center_latitude, center_longitude = lat,lon
                    cur_time = time_sec
                    temp = True
                    for p_lat,p_lon in loc_list:
                        if haversine_distance(lat,lon,p_lat,p_lon) > k_d:
                            temp = False # 距离外
                            break
                    time_gap = cur_time-pre_time
                    if pre_time == 0 or len(loc_list) == 0:
                        loc_list.append((lat,lon))  
                        loc_id_list.append(loc)
                    elif time_gap<5400 and temp == True:
                        loc_list.append((pre_lat,pre_lon))  
                        loc_id_list.append(loc)
                        center_latitude, center_longitude = calculate_center_coordinates(loc_list) 
                    else:
                        session += 1
                        pre_lat,pre_lon = lat,lon
                        loc_list,loc_id_list = [],[]
                        loc_list.append((pre_lat,pre_lon))  
                        loc_id_list.append(loc)

                    pre_lat,pre_lon = lat,lon
                    pre_time = time_sec     
                    session_dic[session] = (user_idx,tw_wday,tw_hour,center_latitude, center_longitude)

                    seq_new.append((user_idx, loc_idx, time_sec, tw_day, tw_wday, tw_hour, lon, lat, session, trajectory_id, cat, SplitTag))
                    cnt += 1
                if cnt > 5:
                    n_users += 1
                    user_seq_array.append(seq_new)
        print(session)
        return user_seq_array,user2idx,n_users,session_dic


    
def calculate_center_coordinates(locations):
    total_latitude = 0
    total_longitude = 0

    for location in locations:
        total_latitude += float(location[0])
        total_longitude += float(location[1])

    center_latitude = total_latitude / len(locations)
    center_longitude = total_longitude / len(locations)

    return center_latitude, center_longitude
