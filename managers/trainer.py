from operator import mod
import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import json
from torch.nn.utils import clip_grad_norm_


class Trainer():
    def __init__(self, params, graph_classifier, train, train_evaluator = None, valid_evaluator=None, test_evaluator = None):
        self.graph_classifier = graph_classifier
        self.train_evaluator=train_evaluator
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train
        self.test_evaluator = test_evaluator
        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        
        logging.info('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)

        self.criterion = nn.BCELoss()
        self.reset_training_state()

        self.all_train_loss = []
        self.all_valid_loss = []
        self.all_test_loss = []

        self.all_train_auc = []
        self.all_valid_auc = []
        self.all_test_auc = []

        self.all_train_aupr = []
        self.all_valid_aupr = []
        self.all_test_aupr = []

        self.all_train_f1_score = []
        self.all_valid_f1_score = []
        self.all_test_f1_score = []

        self.best_test_result = {}


    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def load_model(self):
        self.graph_classifier.load_state_dict(torch.load("my_resnet.pth"))


    def l2_regularization(self, weight):
        l2_loss = []
        for module in self.graph_classifier.modules():
            if type(module) is nn.Linear:
                l2_loss.append((module.weight ** 2).sum() / 2)
        return weight * sum(l2_loss)


    def train_epoch(self):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_preds_scores = []

        train_all_auc = []
        train_all_aupr = []
        train_all_f1 = []

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())
        bar = tqdm(enumerate(dataloader))

        self.params.eval_every_iter = int(self.train_data.num_graphs_pairs/self.params.batch_size) + 1

        for b_idx, batch in bar:
            data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(batch, self.params.device)
            self.optimizer.zero_grad()
            score_pos, g_rep = self.graph_classifier(data_pos)
            
            # BCELoss
            m = nn.Sigmoid()
            score_pos = torch.squeeze(m(score_pos))

            loss_train = self.criterion(score_pos, r_labels_pos)
            model_params = list(self.graph_classifier.parameters())

            l2 = self.l2_regularization(self.params.l2)
            loss = torch.sum(loss_train) + l2

            loss.backward()
            clip_grad_norm_(self.graph_classifier.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            self.updates_counter += 1
            # calculate train metric
            bar.set_description('epoch: ' + str(b_idx+1) + '/ loss_train: ' + str(loss.cpu().detach().numpy()))
            with torch.no_grad():
                total_loss += loss.item()
                target = r_labels_pos.to('cpu').numpy().flatten().tolist()
                all_labels += target

                y_preds = score_pos.cpu().flatten().tolist()
                all_preds += y_preds

                pred_scores = [1 if i else 0 for i in (np.asarray(y_preds) >= 0.5)]
                all_preds_scores += pred_scores

                train_auc = metrics.roc_auc_score(target, y_preds)
                p, r, t = metrics.precision_recall_curve(target, y_preds)
                train_aupr = metrics.auc(r, p)

                train_f1 = metrics.f1_score(target, pred_scores)

                train_all_auc.append(train_auc)
                train_all_aupr.append(train_aupr)
                train_all_f1.append(train_f1)

            # calculate valida and test metric
            if self.updates_counter % self.params.eval_every_iter == 0:
                self.make_valid_test()
            
        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        train_loss = total_loss / b_idx
        train_auc = np.mean(train_all_auc)
        train_aupr = np.mean(train_all_aupr)
        train_f1 = np.mean(train_all_f1)

        return train_loss, train_auc, train_aupr, train_f1, weight_norm


    def train(self):
        self.reset_training_state()

        all_train_loss = []
        all_train_auc = []
        all_train_aupr = []
        all_train_f1_score = []

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            
            train_loss, train_auc, train_aupr, train_f1, weight_norm = self.train_epoch()
            all_train_loss.append(train_loss)
            all_train_auc.append(train_auc)
            all_train_aupr.append(train_aupr)
            all_train_f1_score.append(train_f1)

            self.all_train_loss.append(train_loss)
            self.all_train_auc.append(train_auc)
            self.all_train_aupr.append(train_aupr)
            self.all_train_f1_score.append(train_f1)

            time_elapsed = time.time() - time_start
            logging.info(f'Epoch {epoch} with loss: {train_loss}, training auc: {train_auc}, training aupr: {train_aupr}, weight_norm: {weight_norm} in {time_elapsed}')

            np.save('ke_embed.npy',self.graph_classifier.gnn.embed.cpu().tolist())
            #early stop
            if self.not_improved_count > self.params.early_stop:
                break

        re = [self.all_train_loss, self.all_valid_loss, self.all_test_loss, self.all_train_auc, self.all_valid_auc, self.all_test_auc, self.all_train_aupr, self.all_valid_aupr, self.all_test_aupr, self.all_train_f1_score, self.all_valid_f1_score, self.all_test_f1_score]
        now = time.strftime("%Y-%m-%d-%H_%M",time.localtime(time.time())) 
        np.save(os.path.join(self.params.exp_dir, now + 'result.npy'), np.array(re))


    def make_valid_test(self):
        tic = time.time()
        test_result, test_reps = self.test_evaluator.eval()
        result, reps = self.valid_evaluator.eval()

        logging.info('\033[95m Eval Performance:' + str(result) + 'in ' + str(time.time() - tic)+'\033[0m')
        logging.info('\033[93m Test Performance:' + str(test_result) + 'in ' + str(time.time() - tic)+'\033[0m')
        if result['auc'] >= self.best_metric:
            self.save_classifier()
            self.best_metric = result['auc']
            self.not_improved_count = 0
            self.best_test_result = test_result
            # self.save_representation(test_reps)

            logging.info('\033[93m Test Performance Per Class:' + str(test_result) + 'in ' + str(time.time() - tic)+'\033[0m')
        else:
            self.not_improved_count += 1
            if self.not_improved_count > self.params.early_stop:
                logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                logging.info('\033[93m Test Performance Per Class:' + str(self.best_test_result) + 'in ' + str(time.time() - tic)+'\033[0m')
        
        self.last_metric = result['auc']
    
        valid_loss, valid_auc, valid_aupr, valid_f1_score, test_loss, test_auc, test_aupr, test_f1_score = result['loss'], result['auc'], result['aupr'], result['f1_score'], test_result['loss'], test_result['auc'], test_result['aupr'], test_result['f1_score']

        self.all_valid_loss.append(valid_loss)
        self.all_valid_auc.append(valid_auc)
        self.all_valid_aupr.append(valid_aupr)
        self.all_valid_f1_score.append(valid_f1_score)

        self.all_test_loss.append(test_loss)
        self.all_test_auc.append(test_auc)
        self.all_test_aupr.append(test_aupr)
        self.all_test_f1_score.append(test_f1_score)

    def case_study(self):
        self.reset_training_state()
        self.test_evaluator.print_result(self.params.exp_dir)

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))
        logging.info('Better models found w.r.t accuracy. Saved it!')

    def save_representation(self, best_reps):
        np.savetxt(os.path.join(self.params.exp_dir, 'pair_representation.csv'), best_reps[0], delimiter='\t')
        np.savetxt(os.path.join(self.params.exp_dir, 'pair_pred_label.csv'), best_reps[1], delimiter='\t')
        np.savetxt(os.path.join(self.params.exp_dir, 'pair_true_label.csv'), best_reps[2], delimiter='\t')



