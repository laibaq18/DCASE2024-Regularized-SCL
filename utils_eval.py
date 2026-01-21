import torch
import torch.nn as nn
import torchaudio
from torchaudio import transforms as T
from models import ResNet
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from util import finetune_proto, get_distance, finetune_ce
import glob
import os
from augmentations import RandomCrop, Resize, GaussNoise, Compander, FreqShift
from evaluation import evaluate
from args import args


def evaluate_with_ce(loader_q, encoder, args):
    labels_pred = []
    encoder.eval()
    for x_q in tqdm(loader_q):
        _, z_q = encoder(x_q[0].to(args.device))
        label_pred = torch.argmax(z_q.detach().cpu(), dim=-1)
        labels_pred.extend(label_pred.tolist())
    return labels_pred

def evaluate_with_protoloss(encoder, loader_pos, loader_neg, loader_q, makeview, args, emb_dim):
    encoder.eval()
    with torch.no_grad():
        pos_proto = []
        pos_feat = torch.zeros(0, emb_dim)
        for b_idx, x_p in enumerate(loader_pos):
            if args.multiview:
                zp_views = []
                for i_v in range(args.nviews):
                    xp = x_p[0].to(args.device)
                    if i_v != 0:
                        xp = makeview(xp)
                    zp, _ = encoder(xp)
                    zp = zp.detach().cpu()
                    zp_views.append(zp)
                zp_views = torch.stack(zp_views)
                zp_mean = zp_views.mean(0).mean(0).unsqueeze(0)
                pos_feat = torch.cat((pos_feat, zp_mean), dim=0)
            else:  
                z_pos, _ = encoder(x_p[0].to(args.device))
                z_pos = z_pos.detach().cpu()
                z_pos_mean = z_pos.mean(dim=0).unsqueeze(0)
                pos_feat = torch.cat((pos_feat, z_pos_mean), dim=0)
        pos_proto = pos_feat.mean(dim=0)

        neg_proto = []
        neg_feat = torch.zeros(0, emb_dim)
        for b_idx, x_n in enumerate(loader_neg):
            if args.multiview:
                zn_views = []
                z_feat = torch.zeros(0, emb_dim)
                for i_v in range(args.nviews):
                    xn = x_n[0].to(args.device)
                    if i_v != 0:
                        xn = makeview(xn)
                    zn, _ = encoder(xn)
                    zn = zn.detach().cpu()
                    zn_views.append(zn)
                zn_views = torch.stack(zn_views)
                zn_mean = zn_views.mean(0).mean(0).unsqueeze(0)
                neg_feat = torch.cat([neg_feat, zn_mean], dim=0)            
            else:  
                z_neg, _ = encoder(x_n[0].to(args.device))
                z_neg = z_neg.detach().cpu()
                z_neg_mean = z_neg.mean(dim=0).unsqueeze(0)
                neg_feat = torch.cat((neg_feat, z_neg_mean), dim=0)
        neg_proto = neg_feat.mean(dim=0)

        labels_pred = []
        for x_q in tqdm(loader_q):
            if args.multiview:
                zq_views = []
                for i_v in range(args.nviews):
                    xq = x_q[0].to(args.device)
                    if i_v != 0:
                        xq = makeview(xq)
                    zq, _ = encoder(xq)
                    zq = zq.detach().cpu()
                    zq_views.append(zq)
                zq_views = torch.stack(zq_views)
                z_q = zq_views.mean(0)
            else:
                z_q, _ = encoder(x_q[0].to(args.device))
                z_q = z_q.detach().cpu()
            distances = get_distance(pos_proto, neg_proto, z_q)
            label_pred = torch.argmax(distances, dim=-1)

            labels_pred.extend(label_pred.tolist())
    
    return labels_pred