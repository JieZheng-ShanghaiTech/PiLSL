import pandas as pd
import numpy as np
import random
import os
import scipy.sparse as sp
from sklearn.model_selection import ShuffleSplit, KFold, StratifiedKFold


class SplitDataset():
    def __init__(self, file):
        self.file_path = file

    def SplitByPairs(self):
        df_sl = pd.read_csv(self.file_path, header=None)
        # If no negtive samples, generate them randomly.
        # df_sl = pd.DataFrame(self.generateNSByRandom(df_sl.values.tolist()))
        sl_np_x = df_sl[[0, 1]].to_numpy()
        sl_np_y = df_sl[2].to_numpy()

        # index is the fold
        index = 1

        random_seed = 43
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

        for train_index, test_index in kf.split(sl_np_x, sl_np_y):
            kf_train_valid = StratifiedKFold(n_splits=8, shuffle=True, random_state=random_seed)
            train_x = sl_np_x[train_index]
            train_y = sl_np_y[train_index]
            for train_train_index, train_valid_index in kf_train_valid.split(train_x, train_y):
                train_train_x = train_x[train_train_index]
                train_train_y = train_y[train_train_index].reshape(-1, 1)

                train_valid_x = train_x[train_valid_index]
                train_valid_y = train_y[train_valid_index].reshape(-1, 1)

                train_data = np.concatenate((train_train_x, train_train_y), axis=1)
                valid_data = np.concatenate((train_valid_x, train_valid_y), axis=1)
                break

            test_x = sl_np_x[test_index]
            test_y = sl_np_y[test_index].reshape(-1, 1)
            test_data = np.concatenate((test_x, test_y), axis=1)
            file_path = './C1/cv_' + str(index)
            if os.path.exists(file_path) == False:
                os.makedirs(file_path)
            np.savetxt('./C1/cv_' + str(index) + '/train.txt', train_data, fmt='%d')
            np.savetxt('./C1/cv_' + str(index) + '/dev.txt', valid_data, fmt='%d')
            np.savetxt('./C1/cv_' + str(index) + '/test.txt', test_data, fmt='%d')
            index += 1


    def reindex(self, inter_pairs):
        """reindex function
        Args:
            inter_pairs (triple): 2D
        """
        sl_df = pd.DataFrame(data=inter_pairs)
        set_IDa = set(sl_df[0])
        set_IDb = set(sl_df[1])
        list_all = list(set_IDa | set_IDb)

        orig2id = {}
        id2orig = {}
        for i in range(len(list_all)):
            origin = list_all[i]
            orig2id[origin] = i
            id2orig[i] = origin
        for key in orig2id:
            sl_df.loc[sl_df[0]==key, 0] = orig2id[key]
            sl_df.loc[sl_df[1]==key, 1] = orig2id[key]
        return sl_df.values, orig2id, id2orig, list_all


    def generateNSByRandom(self, inter_pairs):
        all_inters = inter_pairs

        inters_reindex, orig2id, id2orig, gene_list = self.reindex(inter_pairs)
        len_ = len(gene_list)
        edges = inters_reindex.shape[0]
        adj = sp.coo_matrix((np.ones(edges), (inters_reindex[:, 0], inters_reindex[:, 1])), shape=(len_, len_))

        adj_neg = 1 - adj.todense() - np.eye(len_)
        neg_u, neg_v = np.where(adj_neg != 0)
        np.random.seed(43)
        neg_eids = np.random.choice(len(neg_u), edges)
        for neg_idx in range(len(neg_eids)):
            all_inters += [[id2orig[neg_u[neg_eids[neg_idx]]], id2orig[neg_v[neg_eids[neg_idx]]], 0]]

        return np.asarray(all_inters)


    def get_pairs(self, sl_pairs, train_genes, test_genes, type):
        pairs_with_genes = []
        for pair in sl_pairs:
            if type==1:
                if pair[0] in train_genes and pair[1] in train_genes:
                    pairs_with_genes.append(list(pair))
            elif type==2:
                if (pair[0] in test_genes and pair[1] in train_genes) or (pair[0] in train_genes and pair[1] in test_genes):
                    pairs_with_genes.append(list(pair))
            elif type==3:
                if pair[0] in test_genes and pair[1] in test_genes:
                    pairs_with_genes.append(list(pair))
        return pairs_with_genes


    def SplitByGene(self):
        df_sl = pd.read_csv(self.file_path, header=None)
        negative = df_sl[df_sl[2]==0].to_numpy()
        positive = df_sl[df_sl[2]==1].to_numpy()
        gene1 = set(df_sl[0])
        gene2 = set(df_sl[1])
        genes = np.array(list(gene1 | gene2))

        index = 1
        random_seed = 43
        kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

        for train_index, test_index in kf.split(genes):
            train_genes = genes[train_index]
            test_genes = genes[test_index]

            # split train/valid from train gene
            kf_train_valid = KFold(n_splits=8, shuffle=True, random_state=random_seed)
            for train_train_index, train_valid_index in kf_train_valid.split(train_genes):
                train_train_genes = train_genes[train_train_index]
                train_valid_genes = train_genes[train_valid_index]
                break
            
            train_train_positive_pairs = self.get_pairs(positive, train_train_genes, test_genes=None, type=1)
            train_data = self.generateNSByRandom(train_train_positive_pairs)

            train_valid_c2_positive_pairs = self.get_pairs(positive, train_train_genes, test_genes=train_valid_genes, type=2)
            valid_c2_data = self.generateNSByRandom(train_valid_c2_positive_pairs)

            train_valid_c3_positive_pairs = self.get_pairs(positive, train_train_genes, test_genes=train_valid_genes, type=3)
            valid_c3_data = self.generateNSByRandom(train_valid_c3_positive_pairs)

            test_c2_positive_pairs = self.get_pairs(positive, train_train_genes, test_genes, type=2)
            test_c2_data = self.generateNSByRandom(test_c2_positive_pairs)

            test_c3_positive_pairs = self.get_pairs(positive, train_train_genes, test_genes, type=3)
            test_c3_data = self.generateNSByRandom(test_c3_positive_pairs)

            file_path = './C2/cv_' + str(index)
            if os.path.exists(file_path) == False:
                os.makedirs(file_path)
            np.savetxt('./C2/cv_' + str(index) + '/train.txt', train_data, fmt='%d')
            np.savetxt('./C2/cv_' + str(index) + '/dev.txt', valid_c2_data, fmt='%d')
            np.savetxt('./C2/cv_' + str(index) + '/test.txt', test_c2_data, fmt='%d')

            file_path = './C3/cv_' + str(index)
            if os.path.exists(file_path) == False:
                os.makedirs(file_path)
            np.savetxt('./C3/cv_' + str(index) + '/train.txt', train_data, fmt='%d')
            np.savetxt('./C3/cv_' + str(index) + '/dev.txt', valid_c3_data, fmt='%d')
            np.savetxt('./C3/cv_' + str(index) + '/test.txt', test_c3_data, fmt='%d')
            index += 1

if __name__=='__main__':
    # sl_pairs.csv : gene_a, gene_b, lable
    sl_split = SplitDataset('sl_pairs.csv')
    sl_split.SplitByGene()
    sl_split.SplitByPairs()
