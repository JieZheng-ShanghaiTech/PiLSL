from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.dropout = nn.Dropout(p = params.dropout)
        self.relu = nn.ReLU()
        self.train_rels = params.train_rels
        self.relations = params.num_rels
        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)

        # MLP 
        self.mp_layer1 = nn.Linear(self.params.feat_dim, 256)
        self.mp_layer2 = nn.Linear(256, self.params.emb_dim)
        self.bn1 = nn.BatchNorm1d(256)
        
        # Decoder
        if self.params.add_ht_emb and self.params.add_sb_emb:
            if self.params.add_feat_emb and self.params.add_transe_emb:
                self.fc_layer = nn.Linear(3 * (1+self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim) + 2*self.params.emb_dim, 512)
            elif self.params.add_feat_emb :
                self.fc_layer = nn.Linear(3 * (self.params.num_gcn_layers) * self.params.emb_dim + 2*self.params.emb_dim, 512)
            else:
                self.fc_layer = nn.Linear(3 * (1+self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim), 512)
        elif self.params.add_ht_emb:
            self.fc_layer = nn.Linear(2 * (1+self.params.num_gcn_layers) * self.params.emb_dim, 512)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim, 512)
        self.fc_layer_1 = nn.Linear(512, 128)
        self.fc_layer_2 = nn.Linear(128, 1)


    def omics_feat(self, emb):
        self.genefeat = emb

    def get_omics_features(self, ids):
        a = []
        for i in ids:
            a.append(self.genefeat[i.cpu().numpy().item()])
        return np.array(a)


    def forward(self, data):
        g = data
        g.ndata['h'] = self.gnn(g)
        g_out = mean_nodes(g, 'repr')
       
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]
        
        head_feat = torch.FloatTensor(self.get_omics_features(g.ndata['idx'][head_ids])).to(self.params.device)
        tail_feat = torch.FloatTensor(self.get_omics_features(g.ndata['idx'][tail_ids])).to(self.params.device)

        if self.params.add_feat_emb:
            fuse_feat1 = self.mp_layer2( self.bn1(self.relu( self.mp_layer1(head_feat))))
            fuse_feat2 = self.mp_layer2( self.bn1(self.relu( self.mp_layer1(tail_feat))))
            fuse_feat = torch.cat([fuse_feat1, fuse_feat2], dim = 1)


        if self.params.add_ht_emb and self.params.add_sb_emb:
            if self.params.add_feat_emb and self.params.add_transe_emb:
                g_rep = torch.cat([g_out.view(-1, (1+self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim)),
                                   head_embs.view(-1, (1+self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim)),
                                   tail_embs.view(-1, (1+self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim)),
                                   fuse_feat.view(-1, 2*self.params.emb_dim)
                                   ], dim=1)
            elif self.params.add_feat_emb:
                g_rep = torch.cat([g_out.view(-1, (self.params.num_gcn_layers) * self.params.emb_dim),
                                   head_embs.view(-1, (self.params.num_gcn_layers) * self.params.emb_dim),
                                   tail_embs.view(-1, (self.params.num_gcn_layers) * self.params.emb_dim),
                                   fuse_feat.view(-1, 2*self.params.emb_dim)
                                   ], dim=1)
            else:
                g_rep = torch.cat([g_out.view(-1, (1+self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim)),
                                   head_embs.view(-1, (1+self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim)),
                                   tail_embs.view(-1, (1+self.params.num_gcn_layers) * (self.params.emb_dim + self.params.inp_dim)),
                                   #fuse_feat.view(-1, 2*self.params.emb_dim)
                                   ], dim=1)
            
        elif self.params.add_ht_emb:
            g_rep = torch.cat([
                                head_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                tail_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim)
                               ], dim=1)
        else:
            g_rep = g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim)
        
        output = self.fc_layer_2( self.relu(self.fc_layer_1(self.relu(self.fc_layer(self.dropout(g_rep))))))

        return (output, g_rep)
