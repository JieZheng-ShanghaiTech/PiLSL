from torch.utils.data import Dataset
import timeit
import os
import logging
import lmdb
import numpy as np
import json
import pickle
import dgl
from utils.graph_utils import ssp_multigraph_to_dgl, incidence_matrix
from utils.data_utils import process_files, process_files, save_to_file, plot_rel_dist,process_files_decagon
from .graph_sampler import *
import pdb


def generate_subgraph_datasets(params, splits=['train', 'valid', 'test'], saved_relation2id=None, max_label_value=None):
    triple_file = 'data/SynLethKG/kg2id.txt'
       
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel = process_files(params.file_paths, triple_file, saved_relation2id)
    graphs = {}
    
    for split_name in splits:
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}

    for split_name, split in graphs.items():
        split['pairs'] = split['triplets']
    links2subgraphs(adj_list, graphs, params, max_label_value)


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""
    # triple file (KG)
    # entity_type file

    def __init__(self, db_path, db_name, raw_data_paths, included_relations=None, add_traspose_rels=False, use_kge_embeddings=False, dataset='', kge_model='', file_name='', \
        ssp_graph = None,  relation2id= None, id2entity= None, id2relation= None, rel= None,  graph = None, morgan_feat = None):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_name = self.main_env.open_db(db_name.encode())
        self.node_features, self.kge_entity2id = (None, None)
        self.file_name = file_name
        triple_file = 'data/SynLethKG/kg2id.txt'
        self.entity_type = np.loadtxt('data/SynLethKG/entity.txt')

        if not ssp_graph:
            ssp_graph, __, __, __, id2entity, id2relation, rel = process_files(raw_data_paths, triple_file, included_relations)
            self.aug_num_rels = len(ssp_graph)
            self.graph = ssp_multigraph_to_dgl(ssp_graph)
            self.ssp_graph = ssp_graph
        else:
            self.aug_num_rels = len(ssp_graph)
            self.graph = graph
            self.ssp_graph = ssp_graph

        self.num_rels = rel
        self.id2entity = id2entity
        self.id2relation = id2relation

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            # self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            # self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            # self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            # self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            # self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            # self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            # self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            # self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            # self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            # self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            # self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            # self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        logging.info(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")

        with self.main_env.begin(db=self.db_name) as txn:
            self.num_graphs_pairs = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0)

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_name) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)

        return subgraph_pos, g_label_pos, r_label_pos

    def __len__(self):
        return self.num_graphs_pairs

    def _prepare_subgraphs(self, nodes, r_label, n_labels):

        subgraph = dgl.DGLGraph(self.graph.subgraph(nodes))
        #print(subgraph, subgraph.nodes(), subgraph.ndata, subgraph.edges(), subgraph.edata)
        subgraph.edata['type'] = self.graph.edata['type'][self.graph.subgraph(nodes).parent_eid]
        
        subgraph.ndata['idx'] = torch.LongTensor(np.array(nodes))
        subgraph.ndata['ntype'] = torch.LongTensor(self.entity_type[nodes])
        subgraph.ndata['mask'] = torch.LongTensor(np.where(self.entity_type[nodes]==1, 1, 0))
        try:
            edges_btw_roots = subgraph.edge_id(0, 1)
            rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        except AssertionError:
            pass

        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, n_feats)
        # remove interaction
        try:
            edges_btw_roots = subgraph.edge_id(0, 1)
            subgraph.remove_edges(edges_btw_roots)
        except AssertionError:
            pass
        return subgraph #, torch.LongTensor([head_idx, tail_idx])

    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1))
        label_feats[np.arange(n_nodes), n_labels] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids) 

        self.n_feat_dim = n_feats.shape[1]  
        return subgraph

