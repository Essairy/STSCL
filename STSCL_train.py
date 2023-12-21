import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F
from utils import load_model,serialize, unserialize, LadderSampler, generate_square_mask, reset_random_seed, generate_decoder_mask, get_visited_locs,save_model
from tqdm import tqdm
import neg_sampler
from near_loc_query import LocQuerySystem
import loss
import time as Time
from collections import Counter, namedtuple
import yaml
from STSCL import STSCLRec
from stscl_dataset import stscl_dataset
from datautil import collect_fn_step
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import StepLR

from rank_metric import (
    recall,
    ndcg,
    map_k,
    mrr
)


def evaluate(model, test_dataset, negative_sampler, device, batch_size=32, num_neg=100, neg_given=False):
    model.eval()
    reset_random_seed(42)   
    
    loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=lambda e: collect_fn_step(e, negative_sampler, k=num_neg))
    cnt = Counter()
    array = np.zeros(num_neg + 1)
    pred_list = []
    with torch.no_grad():  # user_.t(), loc_.t(), day_.t(),hour_.t(),session_.t(),geo_session_.t(), region_, trg_, batch_trg_regs, trg_nov_, trg_probs_, data_size
        for batch_idx, (user, loc, day, hour, time, lon, lat,  session, pos_category_, pos_hour_, pos_, trg, ds) in enumerate(loader):
            user = user.to(device)
            loc = loc.to(device)
            # time = time.to(device)
            day = day.to(device)
            hour = hour.to(device)
            session = session.to(device)
            # geo_session = geo_session.to(device)
            trg = trg.to(device)
            output = model(user, loc, time, lon, lat, hour, trg ,ds)
            idx = output.sort(descending=True, dim=0)[1]    # 排序后获取相应值在output中的索引
            idx_ = idx.cpu().detach()
            label = trg[0,:].cpu().detach()
            pred_list.append(idx_)
    pred_ = torch.cat(pred_list, dim=1).t()
    label_ = torch.zeros((pred_.shape[0],1))
    recalls, NDCGs, MAPs = {}, {}, {}
    for k_ in (1, 5, 10, 20):
        recalls[k_] = recall(label_, pred_, k_).cpu().detach().numpy().tolist()
        NDCGs[k_] = ndcg(label_, pred_, k_).cpu().detach().numpy().tolist()
        MAPs[k_] = map_k(label_, pred_, k_).cpu().detach().numpy().tolist()
    mrr_res = mrr(label_, pred_).cpu().detach().numpy().tolist()
    for k, v in cnt.items():
        array[k] = v
    return recalls,NDCGs,mrr_res


def train(args , model, train_dataset, test_dataset, valid_dataset, optimizer, scheduler, loss_fn, negative_sampler, test_sampler, device, pretrain_epoch,num_neg=5, batch_size=64, 
          num_epochs=10, batch_size_test=32, num_neg_test=100, test_neg_given=False, num_workers=5):
    epoch_idx = 0
    max_recall5 = 0
    best_epoch = 0
    early_stop = 10
    while epoch_idx<num_epochs:
        start_time = Time.time()
        running_loss = 0.
        processed_batch = 0.
        data_loader = DataLoader(train_dataset, sampler=LadderSampler(train_dataset, batch_size), num_workers=num_workers, batch_size=batch_size, 
                                 collate_fn=lambda e: collect_fn_step(e, negative_sampler, k=num_neg))
        num_batch = len(data_loader)
        print("=====epoch {:>2d}=====".format(epoch_idx + 1))
        batch_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)
        # recalls,NDCGs,mrr_res = evaluate(model, test_dataset, test_sampler, device, config['test']['batch_size'], train_dataset.n_loc-1)
        model.train()
        if epoch_idx == pretrain_epoch:
            optimizer = torch.optim.Adam(model.parameters(), lr=float(0.005), betas=(0.9, 0.98))
        # user_,loc_,wday_,hour_,t_,lon_,lat_,session,cat_,trg_,data_size
        if pretrain_epoch>0 and epoch_idx<=pretrain_epoch and epoch_idx%10 == 0:
            save_model(model, args.save_data_dir, epoch_idx ,pretrain=True)
            print('Save Pretrain!')
        for batch_idx, (user, loc, day, hour, time, lon, lat, session,pos_category_,pos_hour_,pos_,trg, ds) in batch_iterator:
            user = user.to(device)
            loc = loc.to(device)
            day = day.to(device)
            hour = hour.to(device)
            session = session.to(device)
            trg = trg.to(device)
            pos = pos_.to(device)
            pos_category = pos_category_.to(device)
            pos_hour = pos_hour_.to(device)
            lon = lon.to(device)
            lat = lat.to(device)
            optimizer.zero_grad()
            loss1,loss2 = 0,0
            if epoch_idx<pretrain_epoch:
                loss1 = model.pretrain_temporal(args,loc,session)
                if args.intent_loss > 0:
                    loss2 = model.pretrain_intent(user,loc,pos,pos_category,pos_hour,session,lon,lat,ds)
                if epoch_idx % 10 == 0:
                    print('loss_t:',loss1)
                    print('loss_i',loss2)
                loss_pretrain = loss1+loss2
                running_loss += loss_pretrain.item()
                loss_pretrain.backward()
                optimizer.step()
                optimizer.zero_grad()
                processed_batch += 1
                continue
            output1 = model(user, loc, time, lon, lat, hour, trg ,ds)
            output1 = output1.view(-1, loc.size(1), loc.size(0)).permute(2, 1, 0) # (len*(pos+neg)),bs --->(pos+neg)*len*bs--->bs*len*(pos+neg)
            pos_score, neg_score = output1.split([1, num_neg], -1)  
            loss_rec = loss_fn(pos_score, neg_score)

            keep = pad_sequence([torch.ones(e, dtype=torch.float32).to(device) for e in ds], batch_first=True)

            loss_rec = torch.sum(loss_rec * keep) / torch.sum(torch.tensor(ds).to(device))
            # loss2 = model.pretrain_intent(user,loc,pos,pos_category,pos_hour,session,lon,lat)*0.01
            loss_sum = loss_rec
            if epoch_idx % 10 == 0:
                print(loss_rec)
                # print(loss2)
            loss_sum.backward()
            optimizer.step()
            running_loss += loss_sum.item()
            processed_batch += 1
            batch_iterator.set_postfix_str(f"loss={loss_sum.item():.4f}")
        if epoch_idx >= pretrain_epoch:
            scheduler.step()
            current_lr = scheduler.get_last_lr()
            print(f"Current learning rate: {current_lr[0]}")
        epoch_time = Time.time() - start_time
        print("epoch {:>2d} completed.".format(epoch_idx + 1))
        print("time taken: {:.2f} sec".format(epoch_time))
        print("avg. loss: {:.4f}".format(running_loss / processed_batch))
        print("epoch={:d}, loss={:.4f}".format(epoch_idx + 1, running_loss / processed_batch))
        if epoch_idx>pretrain_epoch:
            print("=====evaluation=====valid")
            recalls,NDCGs,mrr_res = evaluate(model, valid_dataset, test_sampler, device, batch_size_test, num_neg_test)
            print("recall@1: {:.4f}, NDCG@1: {:.4f}".format(recalls[1], NDCGs[1]))
            print("recall@5: {:.4f}, NDCG@5: {:.4f}".format(recalls[5], NDCGs[5]))
            print("recall@10: {:.4f}, NDCG@10: {:.4f}".format(recalls[10], NDCGs[10]))
            print("mrr: {:.4f}".format(mrr_res))
            if recalls[5]>max_recall5:
                max_recall5 = recalls[5]
                best_epoch = epoch_idx
                # save_model(model, args.save_dir, epoch_idx, best_epoch)
                t_recalls,t_NDCGs,t_mrr_res = evaluate(model, test_dataset, test_sampler, device, batch_size_test, num_neg_test)
                print("=====evaluation=====test")
                print("recall@1: {:.4f}, NDCG@1: {:.4f}".format(t_recalls[1], t_NDCGs[1]))
                print("recall@5: {:.4f}, NDCG@5: {:.4f}".format(t_recalls[5], t_NDCGs[5]))
                print("recall@10: {:.4f}, NDCG@10: {:.4f}".format(t_recalls[10], t_NDCGs[10]))
                print("recall@20: {:.4f}, NDCG@20: {:.4f}".format(t_recalls[20], t_NDCGs[20]))
                print("mrr: {:.4f}".format(t_mrr_res))
                early_stop = 20
            else:
                early_stop-=1
                if early_stop == 0:
                    print('early stop:',epoch_idx)
                    return
        epoch_idx += 1
    print("training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/base.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data', type=str, default='ca')
    parser.add_argument('--dataset', type=str, default='./dataset/ca')  # foursquare_nyc ca
    parser.add_argument('--dataset_file', type=str, default='ca.csv')
    parser.add_argument('--save_dir', type=str, default='./model_save/ca')
    parser.add_argument('--load_path', type=str, default=False)
    parser.add_argument('--results_path', type=str, default='./result')
    parser.add_argument('--use_exist_data', type=str, default='False')
    parser.add_argument('--use_exit_pretrain', type=str, default='False')
    parser.add_argument('--pretrained_model_name', type=str, default='model_pretrain.pth')
    parser.add_argument('--pretrain_epoch', type=int, default=40)

    parser.add_argument('--user_loss', type=int, default=0)
    parser.add_argument('--spatial_loss', type=int, default=1)
    parser.add_argument('--temporal_loss', type=int, default=1)
    parser.add_argument('--intent_loss', type=int, default=1)

    # lambda
    parser.add_argument('--first_weight', type=float, default=0.001)
    parser.add_argument('--second_weight', type=float, default=0.0005)
    parser.add_argument('--interval_distance', type=float,default=1.5)

    args = parser.parse_args()
    args.trained_model_file = os.path.join(args.save_dir,args.data,args.pretrained_model_name)
    args.save_data_dir = os.path.join(args.save_dir,args.data)
    assert not ((args.use_exit_pretrain == 'True') and (args.pretrain_epoch > 0)),"Dont need pretrain"

    with open(args.config,'r') as file:
        config = yaml.safe_load(file)

    print(config)
    
    filename_raw = os.path.join(args.dataset, args.dataset_file)
    filename_clean = os.path.join(args.dataset, "QuadKeyLSBNDataset.data")  # from GeoSAN code
    loc_query_tree_path = os.path.join(args.dataset, "loc_query_tree.pkl")

    reset_random_seed(222)

    if args.use_exist_data != 'True' or not os.path.isfile(filename_clean):
        print("process data")
        dataset = stscl_dataset(filename_raw, k_d=args.interval_distance)
        filename_clean = os.path.join(args.dataset, str(args.interval_distance)+"QuadKeyLSBNDataset.data")
        serialize(dataset, filename_clean)
        print("save dataset")
    else:
        print('use exist dataset')
        dataset = unserialize(filename_clean)

    train_dataset, test_dataset, valid_dataset = dataset.split(max_len=100)

    if config["train"]["negative_sampler"] in {"PopularitySampler", "KNNPopularitySampler", "KNNSampler", "KNNWRMFSampler"}:
        user_visited_locs = get_visited_locs(dataset)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = STSCLRec(
        nloc=train_dataset.n_loc,
        loc_dim=int(config['model']['location_embedding_dim']),
        nhid=int(config['model']['hidden_dim_encoder']),
        nhead_enc=int(config['model']['num_heads_encoder']),
        num_layers=int(config['model']['num_layers_encoder']),
        session_dic = dataset.session_dic,
        pretrain_weight = args.first_weight,
        intent_weight = args.second_weight
    )
    if args.use_exit_pretrain == 'True':
        print(args.trained_model_file)
        model = load_model(model, args.trained_model_file)
        print('Use pretrained model')

    model.to(device)
    loss_fn = loss.__getattribute__(config['train']['loss'])()

    if config["train"]["negative_sampler"] == "UniformNegativeSampler":
        sampler = neg_sampler.UniformNegativeSampler(train_dataset.n_loc)
    elif config["train"]["negative_sampler"] == "KNNSampler":
        loc_query_sys = LocQuerySystem()
        loc_query_sys.load(loc_query_tree_path)
        sampler = neg_sampler.KNNSampler(
            query_sys=loc_query_sys,
            user_visited_locs=user_visited_locs,
            **config["train"]["negative_sampler_config"]
        )

    if config['optimizer']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config['optimizer']['learning_rate']), betas=(0.9, 0.98))

    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    test_sampler = neg_sampler.UniformallNegativeSampler(
            nloc=train_dataset.n_loc,
        ) 

    print(args)

    train(
        args = args,
        model=model, 
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        valid_dataset = valid_dataset,
        optimizer = optimizer,
        scheduler = scheduler,
        loss_fn=loss_fn,
        negative_sampler=sampler,
        test_sampler=test_sampler,
        device=device,  
        pretrain_epoch = args.pretrain_epoch,
        num_neg=config['train']['num_negative_samples'], 
        batch_size=config['train']['batch_size'],
        num_epochs=config['train']['num_epochs'],
        batch_size_test=config['test']['batch_size'],
        num_neg_test=train_dataset.n_loc-1,
        num_workers=config['train']['num_workers'],
    )