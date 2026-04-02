import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.transformer import Transformer
from transformer.cross_transformer import EncoderLayer
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

    
class MSH_GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, residual=True, num_subset=3):
        super().__init__()
        if not adaptive:
            raise ValueError("Only adaptive graph is supported")
        self.num_layers = A.shape[0]
        self.num_subset = num_subset
        inter_c = out_channels // (num_subset + 1)

        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)

        self.conv_down = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.conv_down.append(nn.Sequential(
                nn.Conv2d(in_channels, inter_c, 1), nn.BatchNorm2d(inter_c), nn.ReLU(True)
            ))
            subsets = nn.ModuleList([
                nn.Sequential(nn.Conv2d(inter_c, inter_c, 1), nn.BatchNorm2d(inter_c))
                for _ in range(num_subset)
            ])
            subsets.append(EdgeConv(inter_c, inter_c, k=3))
            self.conv_layers.append(subsets)

        if residual and in_channels == out_channels:
            self.down = lambda x: x
        elif residual:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: 0

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self.conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                self.bn_init(m, 1)
        self.bn_init(self.bn, 1e-6)

    def forward(self, x):
        out = []
        for i in range(self.num_layers):
            x_down = self.conv_down[i](x)
            y = [self.conv_layers[i][j](torch.einsum('n c t u, v u -> n c t v', x_down, self.PA[i, j]))
                 for j in range(self.num_subset)]
            y.append(self.conv_layers[i][-1](x_down))   # EdgeConv
            out.append(torch.cat(y, dim=1))
        out = torch.stack(out, dim=2).sum(dim=2)        
        out = self.bn(out) + self.down(x)
        return self.relu(out)

    def conv_init(self, conv):
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def bn_init(self, bn, scale):
        nn.init.constant_(bn.weight, scale)
        nn.init.constant_(bn.bias, 0)

class HAF(nn.Module):
    def __init__(self, in_channels, num_layers, num_joints=16, CoM=0):
        super().__init__()
        self.num_layers = num_layers
        self.num_joints = num_joints
        self.CoM = CoM
        self.hierarchy_groups = self.get_hierarchy_groups()
        inter_c = in_channels // 4

        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels, inter_c, 1), nn.BatchNorm2d(inter_c), nn.ReLU(True)
        )
        self.h_edge_conv = EdgeConv(inter_c, inter_c, k=min(3, num_layers))
        self.aggregate = nn.Conv1d(inter_c, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def get_hierarchy_groups(self):
        groups_map = {
            0: [[0], [1,10,13], [2,11,14], [3,4,7,12,15], [5,8], [6,9]],
            1: [[1], [2,0], [3,4,7,10,13], [5,8,11,14], [6,9,12,15]],
            2: [[2], [1,3,4,7], [0,5,8], [6,9,10,13], [11,14], [12,15]]
        }
        if self.CoM in groups_map:
            return groups_map[self.CoM]
        group_size = max(1, self.num_joints // self.num_layers)
        return [list(range(i, min(i+group_size, self.num_joints))) for i in range(0, self.num_joints, group_size)]

    def forward(self, x):
        N, C, L, T, V = x.shape
        x_t = x.mean(dim=-2)                     # [N, C, L, V]
        x_t = self.conv_down(x_t)                # [N, inter_c, L, V]

        sampled = []
        for i in range(L):
            idx = self.hierarchy_groups[i] if i < len(self.hierarchy_groups) else list(range(V))
            sampled.append(x_t[:, :, i, idx].mean(dim=-1, keepdim=True))
        x_sampled = torch.cat(sampled, dim=2)    # [N, inter_c, L]

        att = self.h_edge_conv(x_sampled, dim=3) # [N, inter_c, L]
        att = self.aggregate(att)                # [N, C, L]
        att = self.sigmoid(att).view(N, C, L, 1, 1)

        return (x * att).sum(dim=2)              # [N, C, T, V]


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, dim=4):
        if dim != 3:
            N, C, T, V = x.shape
            x = x.mean(dim=-2)          # [N, C, V]
        else:
            N, C, L = x.shape
            V = L                       
        k = min(self.k, V)
        x = self._get_graph_feature(x, k)
        x = self.conv(x).max(dim=-1)[0] # [N, out_c, V]
        if dim != 3:
            x = repeat(x, 'n c v -> n c t v', t=T)
        return x

    def _knn(self, x, k):
        # x: [N, C, V]
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        dist = -xx - inner - xx.transpose(2, 1)
        return dist.topk(k=k, dim=-1)[1]

    def _get_graph_feature(self, x, k, idx=None):
        N, C, V = x.shape
        if idx is None:
            idx = self._knn(x, k)
        device = x.device
        idx_base = torch.arange(N, device=device).view(-1,1,1) * V
        idx = (idx + idx_base).view(-1)

        x = rearrange(x, 'n c v -> n v c')
        neighbors = rearrange(x, 'n v c -> (n v) c')[idx].view(N, V, k, C)
        x_expand = repeat(x, 'n v c -> n v k c', k=k)

        feat = torch.cat((neighbors - x_expand, x_expand), dim=3)
        return rearrange(feat, 'n v k c -> n c v k')

class Tempoformer(nn.Module):
    def __init__(self, in_channel, out_channel, stride, d_model=512, max_len=100,
                 n_layers=1, n_head=8, drop_prob=0.1, device='cuda',
                 src_pad_idx=0):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(d_model)
        self.transformer = Transformer(src_pad_idx=src_pad_idx,
                                       d_model=d_model,
                                       n_head=n_head,
                                       max_len=max_len,
                                       ffn_hidden=2 * d_model,
                                       n_layers=n_layers,
                                       drop_prob=drop_prob,
                                       device=device)
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        N, C, T, V = x.size()

        x_T = x.permute(0, 3, 1, 2).reshape(N * V, C, T)
        x_T = x_T.permute(0, 2, 1)
        out_T = self.transformer(x_T) 
        out_T = out_T.view(N, V, T, C).permute(0, 3, 2, 1) 
        out_T = self.bn(self.conv(out_T))
        return out_T


class HGT_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, num_joints=16, CoM=0):
        super(HGT_unit, self).__init__()
        self.gcn1 = MSH_GraphConv(
                in_channels=in_channels, 
                out_channels=out_channels,
                A=A,
                adaptive=True,
                residual=True,
                num_subset=3,
            )
        num_layers = A.shape[0] if hasattr(A, 'shape') else 1
        self.aha = HAF(
                    in_channels=out_channels,
                    num_layers=num_layers,
                    num_joints=num_joints,
                    CoM = CoM
                )
        self.transformer = Tempoformer(in_channel=out_channels, out_channel=out_channels, stride=stride,
                                          d_model=out_channels)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1))

    def forward(self, x):
        y = self.gcn1(x)
        if len(y.shape) == 4:
            y = y.unsqueeze(2)
        y = self.aha(y)

        x = self.transformer(y) + self.residual(x)
        return self.relu(x)
    
class MPSTFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.transformer = EncoderLayer(d_model, 8, 2 * d_model, 0.1)
        self.spatial_att_p = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 16, 1, bias=False), nn.ReLU(),
            nn.Conv2d(16, 16, 1, bias=False), nn.Sigmoid()
        )
        self.spatial_att_m = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 16, 1, bias=False), nn.ReLU(),
            nn.Conv2d(16, 16, 1, bias=False), nn.Sigmoid()
        )

    def forward(self, x, y):
        dx, dy = self.transformer(x, y)
        x, y = x + dx, y + dy
        att_p = self.spatial_att_p(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        att_m = self.spatial_att_m(y.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        x, y = x + y * att_m, y + x * att_p
        return x, y
    
class RenovateNet(nn.Module):
    def __init__(self, n_channel, n_class, alp=0.125, tmp=0.125, mom=0.9, h_channel=None,
                 pred_threshold=0.0, use_p_map=True):
        super().__init__()
        self.n_class = n_class
        self.alp, self.tmp, self.mom = alp, tmp, mom
        self.pred_threshold, self.use_p_map = pred_threshold, use_p_map

        self.h_channel = h_channel or n_channel
        self.avg_f = torch.randn(self.h_channel, n_class)
        self.cl_fc = nn.Linear(n_channel, self.h_channel)
        self.loss = nn.CrossEntropyLoss()

    def onehot(self, label):
        return F.one_hot(label, self.n_class).float()

    def get_mask_fn_fp(self, lbl_one, pred_one, logit):
        tp = lbl_one * pred_one
        fn = lbl_one - tp
        fp = pred_one - tp
        tp = tp * (logit > self.pred_threshold).float()
        has_fn = (fn.sum(0) > 1e-8).float().unsqueeze(1)
        has_fp = (fp.sum(0) > 1e-8).float().unsqueeze(1)
        return tp, fn, fp, has_fn, has_fp

    def local_avg_tp_fn_fp(self, f, mask, fn, fp):
        f = f.permute(1, 0)                     # [C, N]
        avg_f = self.avg_f.detach().to(f.device)

        f_fn = torch.matmul(f, F.normalize(fn, p=1, dim=0))
        f_fp = torch.matmul(f, F.normalize(fp, p=1, dim=0))

        mask_sum = mask.sum(0, keepdim=True)
        f_mask = torch.matmul(f, mask) / (mask_sum + 1e-12)
        has_obj = (mask_sum > 1e-8).float()
        has_obj = torch.where(has_obj > 0.1, torch.full_like(has_obj, self.mom), torch.ones_like(has_obj))
        f_mem = avg_f * has_obj + (1 - has_obj) * f_mask
        with torch.no_grad():
            self.avg_f = f_mem
        return f_mem, f_fn, f_fp

    def get_score(self, feature, lbl_one, logit, f_mem, f_fn, f_fp, s_fn, s_fp, mask_tp):
        feature = F.normalize(feature, dim=1)
        f_mem = F.normalize(f_mem.permute(1, 0), dim=1)
        f_fn  = F.normalize(f_fn.permute(1, 0), dim=1)
        f_fp  = F.normalize(f_fp.permute(1, 0), dim=1)

        p_map = ((1 - logit) * lbl_one * self.alp) if self.use_p_map else (lbl_one * self.alp)

        score_mem = torch.matmul(f_mem, feature.T)                     # [K, N]
        score_fn  = torch.matmul(f_fn, feature.T) - 1
        score_fp  = -torch.matmul(f_fp, feature.T) - 1

        fn_map = score_fn * p_map.T * s_fn
        fp_map = score_fp * p_map.T * s_fp

        return (score_mem + fn_map) / self.tmp, (score_mem + fp_map) / self.tmp

    def forward(self, feature, lbl, logit, return_loss=True):
        feature = self.cl_fc(feature)
        pred = logit.argmax(1)
        lbl_one = self.onehot(lbl)
        pred_one = self.onehot(pred)

        logit_soft = logit.softmax(1)
        mask, fn, fp, has_fn, has_fp = self.get_mask_fn_fp(lbl_one, pred_one, logit_soft)
        f_mem, f_fn, f_fp = self.local_avg_tp_fn_fp(feature, mask, fn, fp)
        score_fn, score_fp = self.get_score(feature, lbl_one, logit_soft, f_mem, f_fn, f_fp, has_fn, has_fp, mask)

        if return_loss:
            return (self.loss(score_fn.T, lbl) + self.loss(score_fp.T, lbl)).mean()
        else:
            return score_fn.T.contiguous(), score_fp.T.contiguous()
class PRC(nn.Module):
    def __init__(self, n_channel, n_frame, n_joint, n_person, h_channel=256, **kwargs):
        super(PRC, self).__init__()
        self.n_channel = n_channel
        self.n_frame = n_frame
        self.n_joint = n_joint
        self.n_person = n_person

        self.spatio_cl_net = RenovateNet(n_channel=h_channel // n_joint * n_joint, h_channel=h_channel, **kwargs)
        self.tempor_cl_net = RenovateNet(n_channel=h_channel // n_frame * n_frame, h_channel=h_channel, **kwargs)

        self.spatio_squeeze = nn.Sequential(nn.Conv2d(n_channel, h_channel // n_joint, kernel_size=1),
                                            nn.BatchNorm2d(h_channel // n_joint), nn.ReLU(True))
        self.tempor_squeeze = nn.Sequential(nn.Conv2d(n_channel, h_channel // n_frame, kernel_size=1),
                                            nn.BatchNorm2d(h_channel // n_frame), nn.ReLU(True))

    def forward(self, raw_feat, lbl, logit, **kwargs):
        raw_feat = raw_feat.view(-1, self.n_person, self.n_channel, self.n_frame, self.n_joint)

        spatio_feat = raw_feat.mean(1).mean(-2, keepdim=True)
        spatio_feat = self.spatio_squeeze(spatio_feat)
        spatio_feat = spatio_feat.flatten(1)
        spatio_cl_loss = self.spatio_cl_net(spatio_feat, lbl, logit, **kwargs)

        tempor_feat = raw_feat.mean(1).mean(-1, keepdim=True)
        tempor_feat = self.tempor_squeeze(tempor_feat)
        tempor_feat = tempor_feat.flatten(1)
        tempor_cl_loss = self.tempor_cl_net(tempor_feat, lbl, logit, **kwargs)

        return spatio_cl_loss + tempor_cl_loss
        
class Model(nn.Module):
    def __init__(self, num_class=4, num_point=16, num_constraints=31, graph=None, graph_args=dict(), in_channels_p=3,
                 in_channels_m=8, drop_out=0, cl_mode=None, multi_cl_weights=[1, 1, 1, 1], cl_version='V0', **kwargs):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A,CoM = self.graph.A
        print(A.shape)

        self.cl_mode = cl_mode
        print(cl_mode)
        self.multi_cl_weights = multi_cl_weights
        self.cl_version = cl_version
        print(cl_version)

        self.data_bn_p = nn.BatchNorm1d(in_channels_p * num_point)
        self.data_bn_m = nn.BatchNorm1d(in_channels_m * num_point)

        self.l1_p = HGT_unit(in_channels_p, 64, A, CoM=CoM,residual=False)
        self.l1_m = HGT_unit(in_channels_m, 64, A, CoM=CoM,residual=False)

        self.l2_p = HGT_unit(64, 64, A, CoM=CoM)
        self.l5_p = HGT_unit(64, 128, A, CoM=CoM, stride=2)
        self.l8_p = HGT_unit(128, 256, A, CoM=CoM, stride=2)

        self.l2_m = HGT_unit(64, 64, A, CoM=CoM)
        self.l5_m = HGT_unit(64, 128, A, CoM=CoM, stride=2)
        self.l8_m = HGT_unit(128, 256, A, CoM=CoM, stride=2)

        self.fusion = MPSTFusion(64)

        self.fc1_classifier_p = nn.Linear(256, num_class)
        self.fc1_classifier_m = nn.Linear(256, num_class)
        self.fc2_aff = nn.Linear(256, num_constraints * 48)

        nn.init.normal_(self.fc1_classifier_m.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc1_classifier_p.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc2_aff.weight, 0, math.sqrt(2. / (num_constraints * 48)))
        bn_init(self.data_bn_p, 1)
        bn_init(self.data_bn_m, 1)

        if self.cl_mode is not None:
            self.build_cl_blocks()

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def build_cl_blocks(self):
        if self.cl_mode == "ST-Multi-Level":
            self.ren_low = PRC(64, 48, 16, 1, n_class=4)
            self.ren_mid = PRC(64, 48, 16, 1, n_class=4)
            self.ren_high = PRC(128, 24, 16, 1, n_class=4)
            self.ren_fin = PRC(256, 12, 16, 1, n_class=4)
        else:
            raise KeyError(f"no such Contrastive Learning Mode {self.cl_mode}")

    def get_ST_Multi_Level_cl_output_p(self, x, feat_low, feat_mid, feat_high, feat_fin, label):
        logits = self.fc1_classifier_p(x)
        cl_low = self.ren_low(feat_low, label.detach(), logits.detach())
        cl_mid = self.ren_mid(feat_mid, label.detach(), logits.detach())
        cl_high = self.ren_high(feat_high, label.detach(), logits.detach())
        cl_fin = self.ren_fin(feat_fin, label.detach(), logits.detach())
        cl_loss_p = cl_low * self.multi_cl_weights[0] + cl_mid * self.multi_cl_weights[1] + \
                    cl_high * self.multi_cl_weights[2] + cl_fin * self.multi_cl_weights[3]
        return cl_loss_p

    def get_ST_Multi_Level_cl_output_m(self, x, feat_low, feat_mid, feat_high, feat_fin, label):
        logits = self.fc1_classifier_m(x)
        cl_low = self.ren_low(feat_low, label.detach(), logits.detach())
        cl_mid = self.ren_mid(feat_mid, label.detach(), logits.detach())
        cl_high = self.ren_high(feat_high, label.detach(), logits.detach())
        cl_fin = self.ren_fin(feat_fin, label.detach(), logits.detach())
        cl_loss_m = cl_low * self.multi_cl_weights[0] + cl_mid * self.multi_cl_weights[1] + \
                    cl_high * self.multi_cl_weights[2] + cl_fin * self.multi_cl_weights[3]
        return cl_loss_m
    
    def forward(self, x_p, x_m, label=None, get_cl_loss=False, get_hidden_feat=False, **kwargs):
        N, C_p, T, V, M = x_p.size()
        N, C_m, T, V, M = x_m.size()

        def preprocess(x, in_c, bn):
            x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * in_c, T)
            x = bn(x)
            return x.view(N, M, V, in_c, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, in_c, T, V)

        x_p = preprocess(x_p, C_p, self.data_bn_p)
        x_m = preprocess(x_m, C_m, self.data_bn_m)

        layers_p = [self.l1_p, self.l2_p, self.l5_p, self.l8_p]
        layers_m = [self.l1_m, self.l2_m, self.l5_m, self.l8_m]

        feats_p = []
        feats_m = []
        for lp, lm in zip(layers_p[:2], layers_m[:2]):  
            x_p = lp(x_p)
            x_m = lm(x_m)
            feats_p.append(x_p.clone())
            feats_m.append(x_m.clone())

        x_m, x_p = self.fusion(x_m, x_p)

        for lp, lm in zip(layers_p[2:], layers_m[2:]): 
            x_p = lp(x_p)
            x_m = lm(x_m)
            feats_p.append(x_p.clone())
            feats_m.append(x_m.clone())

        def global_pool(x):
            return x.reshape(N, M, x.size(1), -1).mean(3).mean(1)

        x_p, x_m = global_pool(x_p), global_pool(x_m)

        if get_cl_loss and self.cl_mode == "ST-Multi-Level":
            feat_low_p, feat_mid_p, feat_high_p, feat_fin_p = feats_p
            feat_low_m, feat_mid_m, feat_high_m, feat_fin_m = feats_m
            return (self.fc1_classifier_p(x_p), self.fc2_aff(x_p), self.fc1_classifier_m(x_m),
                    self.get_ST_Multi_Level_cl_output_p(x_p, feat_low_p, feat_mid_p, feat_high_p, feat_fin_p, label),
                    self.get_ST_Multi_Level_cl_output_m(x_m, feat_low_m, feat_mid_m, feat_high_m, feat_fin_m, label))

        return self.fc1_classifier_p(x_p), self.fc2_aff(x_p), self.fc1_classifier_m(x_m)

