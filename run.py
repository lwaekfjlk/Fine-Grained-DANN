from tensorboardX import SummaryWriter
import time
writer = SummaryWriter('log/'+str(int(time.time())))
import argparse
import random
import numpy as np
import torch
import collections
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
import dgl
import os
from pprint import pprint
from utils import load, NeighborSampler
from model import GNN


class Runner(object):
    def __init__(self, params):
        self.p = params
        self.device = torch.device('cpu' if self.p.gpu ==
                                   -1 else f'cuda:{params.gpu}')
        #self.device = torch.device('cpu')
        self.data, self.num_classes, self.num_genes, self.id2label = load(
            self.p)
        #print(self.data['shared_gene_tensor'])
        #print(torch.min(self.data['shared_gene_tensor']))
        #print(torch.max(self.data['shared_gene_tensor']))
        #print(len(self.data['shared_gene_tensor']))
        #print(self.data['human_gene_tensor'])
        #print(len(self.data['human_gene_tensor']))
        #print(self.data['mouse_gene_tensor'])
        #print(len(self.data['mouse_gene_tensor']))
        self.model = GNN(in_feats=self.p.dense_dim,
                         shared_gene=self.data['shared_gene_tensor'],
                         human_gene=self.data['human_gene_tensor'],
                         mouse_gene=self.data['mouse_gene_tensor'],
                         n_hidden=self.p.hidden_dim,
                         n_class=self.num_classes,
                         n_layer=self.p.n_layers,
                         activation=F.relu,
                         dropout=self.p.dropout,
                         weighted=self.p.weighted,
                         device=self.device,
                         gene_num=self.num_genes).to(self.device)
        total_trainable_params = sum(p.numel()
                                     for p in self.model.parameters()
                                     if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=params.lr,
                                          weight_decay=self.p.weight_decay)
        self.domain_criterion = torch.nn.CrossEntropyLoss()
        if self.p.num_neighbors == 0:
            self.num_neighbors = max([
                self.data['mouse']['graph'].number_of_nodes(),
                self.data['human']['graph'].number_of_nodes()
            ])
        else:
            self.num_neighbors = self.p.num_neighbors

        self.data_loader = self.get_dataloader()

    def fit(self):
        max_test_acc = 0
        final_test_report = None
        for epoch in range(self.p.n_epochs):
            # lahelr
            print("Epoch {}".format(epoch))
            start_time = time.time()
            loss = self.train(epoch)
            train_correct, train_total, train_st, _ = self.evaluate('mouse')
            test_correct, test_total, test_st, test_report = self.evaluate(
                'human')
            train_acc, test_acc = train_correct / train_total, test_correct / test_total

            if test_acc > max_test_acc:
                max_test_acc = test_acc
                final_test_report = test_report

            if epoch % 5 == 0:
                print(
                    f"E [{epoch}], loss: {loss:.5f}, train acc: {train_acc:.4f}, test acc: {test_acc:.4f}, cost: {time.time() - start_time:.2f}s"
                )
            writer.add_scalar('Train/loss',loss, epoch)
            writer.add_scalar('Train/acc',train_acc, epoch)
            writer.add_scalar('Test/acc', test_acc, epoch)
                # for i, (key, value) in enumerate(test_st.items()):
                #     print(f"#{i} [{self.id2label[key]}]: {value}, [{value / test_total:.4f}]")
        print(f"MAX TEST ACC: {max_test_acc:.5f}")
        for i, label in enumerate(self.id2label):
            print(
                f"#{i} [{label}] F1-score={final_test_report[label]['f1-score']:.4f}, precision={final_test_report[label]['precision']:.4f}, recall={final_test_report[label]['recall']:.4f}"
            )

    def train(self, epoch, species='mouse'):
        self.model.train()
        tmp_dataloader = self.get_tmp_dataloader()
        len_dataloader = len(tmp_dataloader['mouse'])

        losses = []
        for i, ((source_blocks, source_edges),
                (target_blocks, target_edges)) in enumerate(
                    zip(tmp_dataloader[species], tmp_dataloader['human'])):
            p = float(i + epoch *
                      len_dataloader) / self.p.n_epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # source_input_nodes : neighbour of batch nodes
            source_input_nodes = source_blocks[0].srcdata[dgl.NID]
            # source_seeds : batch nodes
            source_seeds = source_blocks[-1].dstdata[dgl.NID]
            # sourc_batch_labels : torch.size([256]), label of batch_nodes 
            source_batch_input, source_batch_labels, source_batch_seeds = self.to_device(
                species, source_seeds, source_input_nodes)
            source_blocks = [b.to(self.device) for b in source_blocks]

            shared_gene_tensor = self.data['shared_gene_tensor']
            #print("source_batch_seeds : {}".format(source_batch_seeds.shape))
            #print("shared_features : {}".format(shared_features.shape))
            source_batch_shared_or_not_list = []
            for i in range(source_input_nodes.shape[0]):
                if (source_input_nodes[i] in shared_gene_tensor):
                    source_batch_shared_or_not_list.append(1)
                    #print('source_batch_input_grad : {}'.format(source_batch_input[i].grad))
                else:
                    source_batch_shared_or_not_list.append(-1)
            source_batch_shared_or_not = torch.tensor(source_batch_shared_or_not_list,dtype=torch.float)
            #print("source_batch_input : {}".format(source_batch_input.shape))
            #print("source_batch_label : {}".format(source_batch_labels.shape))
            source_class_output, source_domain_output = self.model(
                source_blocks, source_batch_input,
                self.data[species]['weight'], source_edges, alpha)
            label_loss = self.model.cal_loss(source_class_output,
                                             source_batch_labels,
                                             self.p.lbl_smooth)

            target_input_nodes = target_blocks[0].srcdata[dgl.NID]
            target_seeds = target_blocks[-1].dstdata[dgl.NID]
            target_batch_input, target_batch_labels, target_batch_seeds = self.to_device(
                'human', target_seeds, target_input_nodes)
            target_blocks = [b.to(self.device) for b in target_blocks]

            target_batch_shared_or_not_list = []
            for i in range(target_input_nodes.shape[0]):
                if (target_input_nodes[i] in shared_gene_tensor):
                    target_batch_shared_or_not_list.append(1)
                else:
                    target_batch_shared_or_not_list.append(-1)
            # print("target_batch_shared_or_not_list {}".format(len(target_batch_shared_or_not_list)))
            target_batch_shared_or_not = torch.tensor(target_batch_shared_or_not_list,dtype=torch.float)
            _, target_domain_output = self.model(target_blocks,
                                                 target_batch_input,
                                                 self.data['human']['weight'],
                                                 target_edges, alpha)

            domain_label = torch.tensor(
                [0] * source_domain_output.shape[0] +
                [1] * target_domain_output.shape[0]).long().to(self.device)
            domain_loss = self.domain_criterion(
                torch.cat([source_domain_output, target_domain_output]),
                domain_label)

            loss = domain_loss + label_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    def evaluate(self, species):
        self.model.eval()
        total_correct = 0
        label, pred = [], []
        for step, (blocks, edges) in enumerate(self.data_loader[species]):
            input_nodes = blocks[0].srcdata[dgl.NID]
            seeds = blocks[-1].dstdata[dgl.NID]
            batch_input, batch_labels, batch_seeds = self.to_device(species, seeds,
                                                       input_nodes)
            blocks = [b.to(self.device) for b in blocks]

            shared_gene_tensor =  self.data['shared_gene_tensor']
            """"
            batch_shared_or_not_list = []
            for i in range(input_nodes.shape[0]):
                if (input_nodes[i] in shared_gene_tensor):
                    batch_shared_or_not_list.append(1)
                else:
                    batch_shared_or_not_list.append(-1)

            batch_shared_or_not = torch.tensor(batch_shared_or_not_list,dtype=torch.float)
            """
            with torch.no_grad():
                batch_pred, _ = self.model(blocks, batch_input,
                                           self.data[species]['weight'], edges, alpha=1)
            indices = torch.argmax(batch_pred, dim=1)
            label.extend(batch_labels.tolist())
            pred.extend(indices.tolist())
            total_correct += torch.sum(indices == batch_labels).item()

        pred_statistics = dict(collections.Counter(pred))
        report = classification_report(y_true=label,
                                       y_pred=pred,
                                       target_names=self.id2label,
                                       output_dict=True)
        return total_correct, self.data[species][
            'num_cell'], pred_statistics, report

    def to_device(self, species, seeds, input_nodes):
        #print("{} running to_device hot_mat {}".format(species, self.data[species]['hot_mat'].shape))
        #print("{} input_nodes max {} min {}".format(species, torch.max(input_nodes), torch.min(input_nodes)))
        #print("{} seeds       max {} min {}".format(species, torch.max(seeds), torch.min(seeds)))
        #for i in range(len(input_nodes)):
        #    if (input_nodes[i] > 11932):
        #        print(input_nodes[i], (input_nodes[i] in seeds.tolist()))
        #print(len(input_nodes))
        #print(len(seeds))
        #mouse_hot_mat_test = self.data['mouse']['hot_mat'][11932:,:].numpy()
        #human_hot_mat_test = self.data['human']['hot_mat'][11932:,:].numpy()
        #row, col = np.nonzero(mouse_hot_mat_test)
        #print("mouse gene used {}".format(len(set(col))))
        #row, col = np.nonzero(human_hot_mat_test)
        #print("human gene used {}".format(len(set(col))))
        batch_input = self.data[species]['hot_mat'][input_nodes].to(
            self.device)
        batch_labels = self.data[species]['label'][seeds].to(self.device)
        batch_seeds = self.data[species]['hot_mat'][seeds].to(self.device)
        return batch_input, batch_labels, batch_seeds

    def get_dataloader(self):
        data_loader = dict()

        fanouts = [self.num_neighbors] * self.p.n_layers
        for species in ['human', 'mouse']:
            sampler = NeighborSampler(self.data[species]['graph'], fanouts)
            loader = DataLoader(dataset=self.data[species]['seed_id'].numpy(),
                                batch_size=self.p.batch_size,
                                collate_fn=sampler.sample_blocks,
                                shuffle=False,
                                num_workers=os.cpu_count() // 2)
            data_loader[species] = loader
        return data_loader

    def get_tmp_dataloader(self):
        data_loader = dict()
        seed_dict = dict()
        fanouts = [self.num_neighbors] * self.p.n_layers
        # make up length of dataset
        len_diff = len(self.data['human']['seed_id']) - len(
            self.data['mouse']['seed_id'])
        if len_diff > 0:
            seed_dict['mouse'] = self.data['mouse']['seed_id'].numpy()
            seed_dict['human'] = np.random.choice(
                self.data['human']['seed_id'].numpy(),
                len(self.data['mouse']['seed_id']),
                replace=False)
            # seed_dict['human'] = self.data['human']['seed_id'].numpy()
            # seed_dict['mouse'] = np.concatenate([self.data['mouse']['seed_id'].numpy(),
            #                                      np.random.choice(self.data['mouse']['seed_id'].numpy(), len_diff)])
        else:
            seed_dict['human'] = self.data['human']['seed_id'].numpy()
            seed_dict['mouse'] = np.random.choice(
                self.data['mouse']['seed_id'].numpy(),
                len(self.data['human']['seed_id']),
                replace=False)
            # seed_dict['mouse'] = self.data['mouse']['seed_id'].numpy()
            # seed_dict['human'] = np.concatenate([self.data['human']['seed_id'].numpy(),
            #                                      np.random.choice(self.data['human']['seed_id'].numpy(), -len_diff)])

        assert seed_dict['mouse'].shape == seed_dict['human'].shape

        for species in ['human', 'mouse']:
            sampler = NeighborSampler(self.data[species]['graph'], fanouts)
            loader = DataLoader(dataset=seed_dict[species],
                                batch_size=self.p.batch_size,
                                collate_fn=sampler.sample_blocks,
                                shuffle=True,
                                num_workers=os.cpu_count() // 2)
            data_loader[species] = loader
        return data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=10086)
    parser.add_argument("--dropout",
                        type=float,
                        default=0.1,
                        help="dropout probability")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="GPU id, -1 for cpu")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--n_epochs",
                        type=int,
                        default=100,
                        help="number of training epochs")
    parser.add_argument("--dense_dim",
                        type=int,
                        default=400,
                        help="number of hidden gcn units")
    parser.add_argument("--hidden_dim",
                        type=int,
                        default=200,
                        help="number of hidden gcn units")
    parser.add_argument("--n_layers",
                        type=int,
                        default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--num_neighbors", type=int, default=0)
    parser.add_argument("--tissue", type=str, default='Lung')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('--lbl_smooth',
                        type=float,
                        default=0.,
                        help='Label Smoothing')
    parser.add_argument("--weight",
                        type=str,
                        default='yes',
                        choices=['yes', 'no'])
    params = parser.parse_args()
    params.weighted = True if params.weight == 'yes' else False
    pprint(vars(params))

    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)

    runner = Runner(params)
    runner.fit()
