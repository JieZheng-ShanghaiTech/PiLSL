from logging import critical
from multiprocessing import RLock
import os
import numpy as np
import torch
import pdb
import dgl

from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, auc, average_precision_score
# from torch.utils.data.dataset import T
from tqdm import tqdm
class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def print_result(self, exp_dir):
        y_preds = []
        targets = []
        pred_labels = []
        all_loss = 0

        all_idx = []
        all_edges = []
        all_edges_w = []
        g_reps = []

        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                output, g_rep = self.graph_classifier(data_pos)

                # Save pair-wise representation
                g_reps += g_rep.cpu().tolist()
                m = nn.Sigmoid()
                log = torch.squeeze(m(output))

                criterion = nn.BCELoss(reduce=False)
                loss_eval = criterion(log, r_labels_pos)
                loss = torch.sum(loss_eval)

                all_loss += loss.cpu().detach().numpy().item()/len(r_labels_pos)

                target = r_labels_pos.to('cpu').numpy().flatten().tolist()
                targets += target

                y_pred = output.cpu().flatten().tolist()
                y_preds += y_pred
        
                pred_label = [1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)]
                pred_labels += pred_label
    
                batch_graph = dgl.unbatch(data_pos)

                for g in batch_graph:
                    idx = g.ndata['idx'].to('cpu').numpy()
                    edges = g.edges()
                    edges_detach = (edges[0].cpu().numpy(), edges[1].cpu().numpy())
                    edges_w = g.edata['a'].to('cpu').numpy().reshape(1, -1)[0]

                    all_idx.append(idx)
                    all_edges.append(edges_detach)
                    all_edges_w.append(edges_w)

        np.save(os.path.join('case_study/C1/cv_1', 'graph_nodes.npy'), all_idx)
        np.save(os.path.join('case_study/C1/cv_1', 'predict_label.npy'), np.array([targets, pred_labels]))
        np.save(os.path.join('case_study/C1/cv_1', 'graph_edges.npy'), all_edges)
        np.save(os.path.join('case_study/C1/cv_1', 'graph_edges_atten.npy'), all_edges_w)
        with open(os.path.join('case_study/C1/cv_1', 'result.txt'), 'w') as f:
            for (x,y) in zip(targets, pred_labels):
                f.write('%d %d\n'%(x, y))


    def eval(self, save=False):
        y_pred = []
        y_preds = []
        targets = []
        pred_labels = []
        all_auc = []
        all_aupr = []
        all_f1 = []
        all_loss = 0
        g_reps = []
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):

                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
                output, g_rep = self.graph_classifier(data_pos)
                # Save pair-wise representation
                g_reps += g_rep.cpu().tolist()
                m = nn.Sigmoid()
                log = torch.squeeze(m(output))

                criterion = nn.BCELoss(reduce=False)
                loss_eval = criterion(log, r_labels_pos)
                loss = torch.sum(loss_eval)

                all_loss += loss.cpu().detach().numpy().item()/len(r_labels_pos)

                target = r_labels_pos.to('cpu').numpy().flatten().tolist()
                targets += target

                y_pred = output.cpu().flatten().tolist()
                y_preds += y_pred

                auc_ = roc_auc_score(target, y_pred)
                p, r, t = precision_recall_curve(target, y_pred)
                aupr = auc(r, p)
        
                pred_label = [1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)]
                pred_labels += pred_label
                
                f1 = f1_score(target, pred_label)
                all_auc.append(auc_)
                all_aupr.append(aupr)
                all_f1.append(f1)

        return {'loss': all_loss/b_idx, 'auc': np.mean(all_auc), 'aupr': np.mean(all_aupr), 'f1_score': np.mean(all_f1)}, (g_reps, pred_labels, targets)