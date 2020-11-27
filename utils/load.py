import argparse

import pandas as pd
import dgl
import time
import torch
import torch.nn.functional as F
import collections
import metis
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import PCA
from pathlib import Path
import numpy as np
import networkx as nx
from pprint import pprint
import json


def normalize_weight(g: dgl.DGLHeteroGraph, weight):
    graph = g.local_var()
    graph.edata['weight'] = weight
    in_deg = graph.in_degrees(range(graph.number_of_nodes())).float().unsqueeze(-1)
    graph.ndata['in_deg'] = in_deg
    graph.update_all(dgl.function.copy_edge('weight', 'edge_w'), dgl.function.sum('edge_w', 'total'),
                     lambda nodes: {'norm': nodes.data['total'] / nodes.data['in_deg']})
    graph.apply_edges(lambda edges: {'weight': edges.data['weight'] / edges.dst['norm']})
    return graph.edata['weight']

def get_id_2_gene(data_path: Path, tissue):
    path_dict = {'mouse': data_path / 'mouse', 'human': data_path / 'human'}
    statistic_path = data_path / 'statistic'
    gene_statistic_path = statistic_path / f'{tissue}_genes.txt'
    genes = None
    if not gene_statistic_path.exists():
        for species in ['mouse', 'human']:
            data_files = path_dict[species].glob(f'{species}_clean_{tissue}*_data.csv')
            for file in data_files:
                data = pd.read_csv(file, dtype=np.str, header=0).values[:, 0]
                if genes is None:
                    genes = set(data)
                else:
                    genes = genes & set(data)
                print(file, len(genes), len(set(data)))
        id2gene = list(genes)
        id2gene.sort()
        with open(gene_statistic_path, 'w', encoding='utf-8') as f:
            for gene in id2gene:
                f.write(gene + '\r\n')
    else:
        id2gene = []
        with open(gene_statistic_path, 'r', encoding='utf-8') as f:
            for line in f:
                id2gene.append(line.strip())
    return id2gene


def get_id_2_label(data_path: Path, tissue):
    path_dict = {'mouse': data_path / 'mouse', 'human': data_path / 'human'}
    statistic_path = data_path / 'statistic'
    label_statistic_path = statistic_path / f'{tissue}_labels.txt'
    labels = None
    if not label_statistic_path.exists():
        for species in ['mouse', 'human']:
            data_files = path_dict[species].glob(f'{species}_{tissue}*_celltype.csv')
            for file in data_files:
                df = pd.read_csv(file, dtype=np.str, header=0)
                df['Cell_type'] = df['Cell_type'].map(str.strip)
                if labels is None:
                    labels = set(df.values[:, 2])
                else:
                    labels = set(df.values[:, 2]) & labels
        id2label = list(labels)
        with open(label_statistic_path, 'w', encoding='utf-8') as f:
            for cell_type in id2label:
                f.write(cell_type + '\r\n')
    else:
        id2label = []
        with open(label_statistic_path, 'r', encoding='utf-8') as f:
            for line in f:
                id2label.append(line.strip())
    return id2label


def load(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    tissue = params.tissue
    # device = torch.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}')

    proj_path = Path(__file__).parent.resolve().parent.resolve()
    id2gene = get_id_2_gene(proj_path / 'data', tissue)
    id2label = get_id_2_label(proj_path / 'data', tissue)
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
    label2id = {label: idx for idx, label in enumerate(id2label)}
    num_genes, num_labels = len(id2gene), len(id2label)
    print(f"Totally {num_genes} genes, {num_labels} labels.")
    gene_ids = torch.arange(num_genes, dtype=torch.int32).unsqueeze(-1)
    data = dict()
    human_gene_set = set()
    mouse_gene_set = set()
    shared_gene_set = set()
    for species in ['human', 'mouse']:
        total_node_num = num_genes
        src_list, dst_list = [], []
        weight_list = []
        labels = []
        matrices = []
        num_cells = 0
        species_path = proj_path / 'data' / species
        data_files = species_path.glob(f'*{species}_clean_{tissue}*_data.csv')
        for data_file in data_files:
            start_time = time.time()
            number = ''.join(list(filter(str.isdigit, data_file.name)))
            type_file = species_path / f'{species}_{tissue}{number}_celltype.csv'

            # load celltype file then update labels accordingly
            cell2type = pd.read_csv(type_file, index_col=0)
            cell2type.columns = ['cell', 'type']
            cell2type['type'] = cell2type['type'].map(str.strip)
            cell2type['id'] = cell2type['type'].map(label2id)
            # filter out cells not in label-text
            filter_cell = np.where(pd.isnull(cell2type['id']) == False)[0]
            cell2type = cell2type.iloc[filter_cell]

            assert not cell2type['id'].isnull().any(), 'something wrong about celltype file.'
            labels += cell2type['id'].tolist()

            # load data file then update graph
            df = pd.read_csv(data_file, index_col=0)  # (gene, cell)
            df = df.transpose(copy=True)  # (cell, gene)
            # filter out cells not in label-text
            df = df.iloc[filter_cell]
            assert cell2type['cell'].tolist() == df.index.tolist()
            df = df.rename(columns=gene2id)
            # filter out useless columns if exists (when using gene intersection)
            col = [c for c in df.columns if c in gene2id.values()]
            df = df[col]

            print(f'{data_file.name} Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%')
            # maintain inter-datasets index for graph and RNA-seq values
            arr = df.to_numpy()
            row_idx, col_idx = np.nonzero(arr > params.threshold)  # intra-dataset index
            non_zeros = arr[(row_idx, col_idx)]  # non-zero values
            # cell_idx = row_idx + graph.number_of_nodes()  # cell_index

            """ the index [0, total_node_num] represents the genes  """
            """ the index > total_node_num represents the cells """
            cell_idx = row_idx + total_node_num  # cell_index
            gene_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
            # if species == 'mouse':
            info_shape = (len(df), num_genes)
            info = csr_matrix((non_zeros, (row_idx, gene_idx)), shape=info_shape)
            matrices.append(info)

            num_cells += len(df)
            total_node_num += len(df)
            # add edges
            src_list += cell_idx.tolist() + gene_idx
            dst_list += gene_idx + cell_idx.tolist()
            weight_list.append(torch.tensor(non_zeros, dtype=torch.float32).unsqueeze(1))
            weight_list.append(torch.tensor(non_zeros, dtype=torch.float32).unsqueeze(1))

            print(
                f'{species} -> Added {len(df)} cell nodes and {len(cell_idx)} edges, Time: {time.time() - start_time:.2f}s.')
            # break

            if (species == 'human'):
                human_gene_set = human_gene_set.union(set(gene_idx))
            if (species == 'mouse'):
                mouse_gene_set = mouse_gene_set.union(set(gene_idx))
        sub_data = dict()
        sub_data['graph'] = dgl.graph((src_list, dst_list))
        sub_data['num_cell'] = num_cells
        sub_data['label'] = list(map(int, labels))
        sub_data['matrices'] = matrices
        sub_data['weight'] = torch.cat(weight_list)
        data[species] = sub_data
        assert len(labels) == num_cells

    print("human gene type : {}".format(len(human_gene_set)))
    print("mouse gene type : {}".format((mouse_gene_set)))
    shared_gene_set = human_gene_set
    shared_gene_set = shared_gene_set.intersection(mouse_gene_set)
    print("Human and Mouse share {} genes".format(len(shared_gene_set)))


    for species in ['mouse', 'human']:
        statistics = dict(collections.Counter(data[species]['label']))
        print(f'------{species} label statistics------')
        for i, (key, value) in enumerate(statistics.items(), start=1):
            print(f"#{i} [{id2label[key]}]: {value}, {value / data[species]['num_cell']:.4f}")

    # 2. create features
    sparse_feat = vstack(data['mouse']['matrices']).toarray()  # cell-wise  (cell, gene)
    assert sparse_feat.shape[0] == num_cells
    # transpose to gene-wise
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat.T)
    gene_feat = gene_pca.transform(sparse_feat.T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')

    print("gene_feat: {}".format(gene_feat.shape))
    gene_feat = torch.from_numpy(gene_feat)  # use shared storage
    for species in ['mouse', 'human']:
        sparse_feat = vstack(data[species]['matrices']).toarray()
        # do normalization
        sparse_feat = sparse_feat / (np.sum(sparse_feat, axis=1, keepdims=True) + 1e-8)
        # use weighted gene_feat as cell_feat
        cell_feat = sparse_feat.dot(gene_feat)
        cell_feat = torch.from_numpy(cell_feat)
        features = torch.cat([gene_feat, cell_feat], dim=0).type(torch.float)
        features = (features - torch.mean(features, dim=0)) / torch.sqrt(torch.var(features, dim=0))
        data[species]['features'] = features
        data[species]['weight'] = normalize_weight(data[species]['graph'], data[species]['weight'])
        print(
            f"{species}: #NODES: {data[species]['graph'].number_of_nodes()}, #EDGES: {data[species]['graph'].number_of_edges()}, #CELLS: {data[species]['num_cell']}")

        # # add self-loop
        # src, dst = data[species]['graph'].all_edges(order='eid')
        # data[species]['graph'] = dgl.graph((torch.cat([src, torch.arange(data[species]['graph'].number_of_nodes())]),
        #                                     torch.cat([dst, torch.arange(data[species]['graph'].number_of_nodes())])))
        # data[species]['weight'] = torch.cat([data[species]['weight'],
        #                                      torch.ones(data[species]['graph'].number_of_nodes(),
        #                                                 dtype=torch.float).unsqueeze(1)])
        # data[species]['node_id'] = torch.cat(
        #     [gene_ids, torch.tensor([-1] * data[species]['num_cell'], dtype=torch.int32).unsqueeze(-1)])

        data[species]['label'] = torch.tensor([-1] * num_genes + data[species]['label'])
        data[species]['seed_id'] = torch.arange(num_genes, num_genes + data[species]['num_cell'])
        data[species].pop('matrices')

        assert data[species]['features'].shape[0] == data[species][
            'graph'].number_of_nodes(), f"{data[species]['features'].shape[0]} != {data[species]['graph'].number_of_nodes()}"
        assert data[species]['weight'].shape[0] == data[species][
            'graph'].number_of_edges(), f"{data[species]['weight'].shape[0]} != {data[species]['graph'].number_of_edges()}"
        # assert data[species]['features'].shape[0] == data[species]['node_id'].shape[
        #     0], f"{data[species]['features'].shape[0]} != {data[species]['node_id'].shape[0]}"

    return data, len(id2label), num_genes, id2label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=10086)
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=2,
                        help="GPU id, -1 for cpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--n_epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--dense_dim", type=int, default=400,
                        help="number of hidden gcn units")
    parser.add_argument("--hidden_dim", type=int, default=200,
                        help="number of hidden gcn units")
    parser.add_argument("--n_layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--num_neighbors", type=int, default=0)
    parser.add_argument("--tissue", type=str, default='Lung')
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--partition_size", type=int, default=4000)

    params = parser.parse_args()
    load(params)

    # g_hete = dgl.graph(([0, 1, 2, 1, 3, 1, 4, 5], [1, 0, 1, 2, 1, 3, 5, 4]))
    # weight = torch.tensor([1, 1, 2, 2, 3, 3, 5, 5], dtype=torch.float)
    # weight_hete = normalize_weight(g_hete, weight)
    # print(weight_hete)
    # g_homo = dgl.DGLGraph()
    # g_homo.add_nodes(7)
    # g_homo.add_edges([0, 1, 2, 1, 3, 1, 4, 5], [1, 0, 1, 2, 1, 3, 5, 4])
    # weight_homo = normalize_weight(g_homo, weight)
    # print(weight_homo)
