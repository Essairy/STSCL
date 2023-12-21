import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
from layers import TransformerEncoder
from sklearn.cluster import KMeans

def haversine_distance(lat1, lon1, lat2, lon2):
    # 将角度转换为弧度
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])

    # 计算差值
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # 使用球面余弦定理计算距离
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    # 地球半径（单位：公里）
    radius = 6371.0

    # 计算距离
    distance = radius * c

    return distance

class STSCLRec(nn.Module):
    # def __init__(self, nuser, nloc, ntime, nreg, user_dim, loc_dim, time_dim, reg_dim, nhid, nhead_enc, nhead_dec, nlayers, dropout=0.5, **extra_config):
    def __init__(self, nloc, loc_dim,nhid,nhead_enc,session_dic,distance_factor=10,max_seq_length=120,num_layers=1,hidden_dropout_prob=0.5,attn_dropout_prob=0.2,hidden_act = 'gelu',layer_norm_eps = 1e-12,initializer_range=0.02,pretrain_weight = 0.001,intent_weight = 0.0005,tau = 0.05):
        super(STSCLRec, self).__init__()
        self.n_layers =num_layers
        self.n_heads = nhead_enc
        self.hidden_size = loc_dim
        self.inner_size = nhid
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_dropout_prob = attn_dropout_prob
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.n_loc = nloc
        self.initializer_range = initializer_range

        self.item_embedding = nn.Embedding(nloc, self.hidden_size, padding_idx=0)

        self.position_embedding = nn.Embedding(max_seq_length, self.hidden_size)

        self.trm_encoder = TransformerEncoder(n_layers=self.n_layers,
                                              n_heads=self.n_heads,
                                              hidden_size=self.hidden_size,
                                              inner_size=self.inner_size,
                                              hidden_dropout_prob=self.hidden_dropout_prob,
                                              attn_dropout_prob=self.attn_dropout_prob,
                                              hidden_act=self.hidden_act,
                                              layer_norm_eps=self.layer_norm_eps)

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.weight = pretrain_weight
        self.intent_weight = intent_weight
        self.tau = tau
        self.session_feature = session_dic

        self.distance_factor = distance_factor
    
    # user,loc,pos,pos_category,pos_hour,session,lon,lat,ds
    def pretrain_intent(self,user,loc,pos,pos_cat,pos_hour,session,lon,lat,ds):
        item_seq = loc
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        
        item_emb = self.item_embedding(item_seq)    # b*l*dim
        input_emb = item_emb  + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1] # bs*l*dim

        pos = pos.squeeze(-1)
        category_CL_loss = self.category_CL_local(user,item_seq,output,pos,pos_cat,pos_hour)

        return category_CL_loss*self.intent_weight

    
    def forward(self, src_user, src_loc, time, lon, lat, hour, trg_loc, ds=None):

        item_seq = src_loc

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        
        item_emb = self.item_embedding(item_seq)    # b*l*dim
        input_emb = item_emb  + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        
        out = output.permute(1,0,2)
        loc_emb_trg = self.item_embedding(trg_loc)

        if self.training:
            src = out.repeat(loc_emb_trg.size(0) // out.size(0), 1, 1)
        else:
            a = torch.tensor(ds) - 1
            b = torch.arange(len(ds))
            src = out[a, b, :]   # 获取末尾
            src = src.unsqueeze(0).repeat(loc_emb_trg.size(0),1, 1) 

        output = torch.sum(src * loc_emb_trg, dim=-1)
        return output
    
    # somecode copy from tcpsrec
    def pretrain_temporal(self,args,src_loc,session):
        loss_cl = 0
        # self.emb_temporal.weight.data.copy_(self.item_embedding.weight.data)
        # self.emb_temporal.to(src_loc.device)
        item_embs = self.item_embedding(src_loc)
        session_mean,session_labels = self.get_mean_of_item(item_embs,session)
        if args.user_loss > 0:
            item_CL_seq_loss = self.item_CL_global(src_loc, item_embs)
            loss_cl += self.weight * item_CL_seq_loss

        if args.spatial_loss > 0:
            item_CL_subseq_loss = self.item_CL_local(src_loc, item_embs, session, session_mean, session_labels)
            loss_cl += self.weight * item_CL_subseq_loss

        if args.temporal_loss > 0:
            subseq_CL_seq_loss = self.subseq_CL_alone(session_mean, session_labels)
            loss_cl += self.weight * subseq_CL_seq_loss
            subseq_CL_subseq_loss = self.subseq_CL_cross(session_mean, session_labels)
            loss_cl += self.weight * subseq_CL_subseq_loss

        return loss_cl 

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

    def get_mean_of_item(self, item_embs, session_seq):
        item_embs = item_embs.reshape(-1, item_embs.shape[-1])
        session_seq = session_seq.reshape(-1)
        session_mean, session_labels = utils.groupby_mean(item_embs, session_seq, device=item_embs.device)
        return session_mean, session_labels
    
    def ssl_loss(self, anchor_embedding, positive_embedding, negative_embedding=None, all_embedding=None):
        if all_embedding is None:
            all_embedding = torch.cat((positive_embedding, negative_embedding), 0)

        norm_anchor_embedding = F.normalize(anchor_embedding)
        norm_positive_embedding = F.normalize(positive_embedding)
        norm_all_embedding = F.normalize(all_embedding)

        pos_score = torch.mul(norm_anchor_embedding, norm_positive_embedding).sum(dim=1)
        ttl_score = torch.matmul(norm_anchor_embedding, norm_all_embedding.transpose(0, 1))
        pos_score = torch.exp(pos_score / self.tau)
        ttl_score = torch.exp(ttl_score / self.tau).sum(dim=1)

        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss

    def category_CL_local(self,user,item_seq,item_hidden,pos,cat,hour):
        real_item_mask = item_seq != 0 
        real_intent_hidden = torch.masked_select(item_hidden, real_item_mask.unsqueeze(2)).reshape(-1, item_hidden.shape[-1])

        real_hour = torch.masked_select(hour, real_item_mask).reshape(-1)
        real_category = torch.masked_select(cat, real_item_mask).reshape(-1)
        # real_target =  torch.masked_select(pos, real_item_mask).reshape(-1)
        real_user = torch.masked_select(user, real_item_mask).reshape(-1)
        intent_feature =real_user*100000+real_category*100+real_hour//6.0    
        # intent_feature =real_user*10000+real_target 

        # cluster_mean_long, cluster_labels_long = utils.groupby_mean(real_intent_hidden, intent_feature, device=real_intent_hidden.device)
        cluster_mean_long, cluster_labels_long = utils.groupby_last(real_intent_hidden, intent_feature, device=real_intent_hidden.device)
        

        cluster_label2idx_long = {l: i for i, l in enumerate(cluster_labels_long.tolist())}
        cluster_embs_idx_long = [cluster_label2idx_long[i] for i in intent_feature.tolist()]
        cluster_embs_long = cluster_mean_long[cluster_embs_idx_long]

        intent_CL_loss = self.ssl_loss(anchor_embedding=real_intent_hidden,
                                                  positive_embedding=cluster_embs_long,
                                                  all_embedding=cluster_mean_long)
        return intent_CL_loss

    def item_CL_global(self, item_seq, item_embs):
        real_item_mask = item_seq != 0  # torch.Size([1024, 50])
        real_item_embs = torch.masked_select(item_embs, real_item_mask.unsqueeze(2)).reshape(-1, item_embs.shape[-1])

        sequence_sum = torch.sum(item_embs * real_item_mask.float().unsqueeze(2), dim=1)  # torch.Size([1024, 64])  序列求和
        sequence_mean = sequence_sum / torch.sum(real_item_mask, dim=1, keepdim=True)  # torch.Size([1024, 64]) 求平均
        sequence_embs_idx = torch.nonzero(real_item_mask)[:, 0]  # torch.Size([X,]) # 获取每个元素所在的行
        sequence_embs = sequence_mean[sequence_embs_idx]  # torch.Size([X, 64])

        item_CL_global_loss = self.ssl_loss(anchor_embedding=real_item_embs,
                                            positive_embedding=sequence_embs,
                                            all_embedding=sequence_mean)
        return item_CL_global_loss

    def item_CL_local(self, item_seq, item_embs, session_seq, session_mean, session_labels):
        real_item_mask = item_seq != 0  # torch.Size([1024, 50])
        real_item_embs = torch.masked_select(item_embs, real_item_mask.unsqueeze(2)).reshape(-1, item_embs.shape[-1])

        session_label2idx = {l: i for i, l in enumerate(session_labels.tolist())}
        real_session_seq = torch.masked_select(session_seq, real_item_mask)
        session_embs_idx = [session_label2idx[i] for i in real_session_seq.tolist()]
        session_embs = session_mean[session_embs_idx]

        item_CL_local_loss = self.ssl_loss(anchor_embedding=real_item_embs,
                                           positive_embedding=session_embs,
                                           all_embedding=session_mean)
        return item_CL_local_loss

    def subseq_CL_alone(self, session_mean, session_labels, geo=False):
        tensor_list = []
        if geo==False:
            for x in session_labels:
                cur_feature = torch.tensor(self.session_feature[x.item()],device=session_mean.device)
                tensor_list.append(cur_feature)
        else:
            for x in session_labels:
                cur_feature = torch.tensor(self.geo_feature[x.item()],device=session_mean.device)
                tensor_list.append(cur_feature)
        session_feature = torch.stack(tensor_list,dim=0)
        # long periodicity
        session_feature_long = session_feature[:, 0] * 10 + session_feature[:, 1]
        cluster_mean_long, cluster_labels_long = utils.groupby_mean(session_mean, session_feature_long, device=session_mean.device)

        cluster_label2idx_long = {l: i for i, l in enumerate(cluster_labels_long.tolist())}
        cluster_embs_idx_long = [cluster_label2idx_long[i] for i in session_feature_long.tolist()]
        cluster_embs_long = cluster_mean_long[cluster_embs_idx_long]

        subseq_CL_alone_loss_long = self.ssl_loss(anchor_embedding=session_mean,
                                                  positive_embedding=cluster_embs_long,
                                                  all_embedding=cluster_mean_long)

        # short periodicity
        session_feature_short = session_feature[:, 0] * 10 + session_feature[:, 2]//6.0
        cluster_mean_short, cluster_labels_long = utils.groupby_mean(session_mean, session_feature_short, device=session_mean.device)

        cluster_label2idx_short = {l: i for i, l in enumerate(cluster_labels_long.tolist())}
        cluster_embs_idx_short = [cluster_label2idx_short[i] for i in session_feature_short.tolist()]
        cluster_embs_short = cluster_mean_short[cluster_embs_idx_short]

        subseq_CL_alone_loss_short = self.ssl_loss(anchor_embedding=session_mean,
                                                   positive_embedding=cluster_embs_short,
                                                   all_embedding=cluster_mean_short)

        return subseq_CL_alone_loss_long + subseq_CL_alone_loss_short

    def subseq_CL_cross(self, session_mean, session_labels,geo=False):
        tensor_list = []
        if geo==False:
            for x in session_labels:
                cur_feature = torch.tensor(self.session_feature[x.item()],device=session_mean.device)
                tensor_list.append(cur_feature)
        else:
            for x in session_labels:
                cur_feature = torch.tensor(self.geo_feature[x.item()],device=session_mean.device)
                tensor_list.append(cur_feature)
        session_feature = torch.stack(tensor_list,dim=0)

        session_feature = session_feature[:, 0] * 100 + session_feature[:, 1] * 10 + session_feature[:, 2]//6.0
        cluster_mean, cluster_labels = utils.groupby_mean(session_mean, session_feature, device=session_mean.device)

        cluster_label2idx = {l: i for i, l in enumerate(cluster_labels.tolist())}
        cluster_embs_idx = [cluster_label2idx[i] for i in session_feature.tolist()]
        cluster_embs = cluster_mean[cluster_embs_idx]

        subseq_CL_cross_loss = self.ssl_loss(anchor_embedding=session_mean,
                                             positive_embedding=cluster_embs,
                                             all_embedding=cluster_mean)
        return subseq_CL_cross_loss
