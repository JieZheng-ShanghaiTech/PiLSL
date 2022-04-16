"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RGCNBasisLayer as RGCNLayer
import numpy as np

from .aggregators import SumAggregator, MLPAggregator, GRUAggregator


class RGCN(nn.Module):
    def __init__(self, params):
        super(RGCN, self).__init__()

        self.max_label_value = params.max_label_value
        self.inp_dim = params.inp_dim
        self.emb_dim = params.emb_dim
        self.attn_rel_emb_dim = params.attn_rel_emb_dim
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.dropout
        self.edge_dropout = params.edge_dropout
        self.has_attn = params.has_attn
        self.num_nodes = params.num_nodes
        self.device = params.device
        self.add_transe_emb = params.add_transe_emb

        if self.has_attn:
            self.attn_rel_emb = nn.Embedding(self.aug_num_rels, self.attn_rel_emb_dim, sparse=False)
        else:
            self.attn_rel_emb = None
        
        # to incorporate the KG embeddings, you need to modify the code here and insert the KG embeddings
        if params.use_kge_embeddings:
            kg_embed = np.load('data/SynLethKG/kg_embedding/kg_TransE_l2_entity.npy')
            self.embed = torch.FloatTensor(kg_embed).to(params.device)
        else:
            self.embed = nn.Parameter(torch.Tensor(self.num_nodes, self.emb_dim), requires_grad = True)
            nn.init.xavier_uniform_(self.embed,
                                    gain=nn.init.calculate_gain('relu'))

        # initialize aggregators for input and hidden layers
        if params.gnn_agg_type == "sum":
            self.aggregator = SumAggregator(self.emb_dim)
        elif params.gnn_agg_type == "mlp":
            self.aggregator = MLPAggregator(self.emb_dim)
        elif params.gnn_agg_type == "gru":
            self.aggregator = GRUAggregator(self.emb_dim)

        # create rgcn layers
        self.build_model()


    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers - 1):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

    def build_input_layer(self):
        return RGCNLayer(self.inp_dim+self.emb_dim,
                         self.inp_dim+self.emb_dim,
                         self.aggregator,
                         self.attn_rel_emb_dim,
                         self.aug_num_rels,
                         self.num_bases,
                         embed = self.embed,
                         num_nodes= self.num_nodes,
                         activation=F.relu,
                         dropout=self.dropout,
                         edge_dropout=self.edge_dropout,
                         is_input_layer=True,
                         has_attn=self.has_attn,
                         add_transe_emb=self.add_transe_emb,
                         one_attn = True)

    def build_hidden_layer(self, idx):
        return RGCNLayer(
                     self.inp_dim+self.emb_dim,
                     self.inp_dim+self.emb_dim,
                     self.aggregator,
                     self.attn_rel_emb_dim,
                     self.aug_num_rels,
                     self.num_bases,
                     embed = self.embed,
                     activation=F.relu,
                     dropout=self.dropout,
                     edge_dropout=self.edge_dropout,
                     has_attn=self.has_attn,
                     add_transe_emb=self.add_transe_emb,
                     one_attn = True)

    def forward(self, g):
        for layer in self.layers:
            layer(g, self.attn_rel_emb)
        return g.ndata.pop('h')
