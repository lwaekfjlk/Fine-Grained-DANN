import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import dgl
import dgl.function as fn

class FineGrainedLayerF(Function):
    @staticmethod
    def forward(ctx, x, y, z):
        res = x+y+z
        return res.view_as(res)

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output.shape)
        return grad_output, grad_output, grad_output


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        #print("backward {}".format(grad_output.shape))
        #shared_pos = torch.where(ctx.shared_or_not<0)
        #shared_pos = shared_pos[0]
        #print('shared_pos : {}'.format(shared_pos))
        #print("origin grad_output : {}".format(grad_output[shared_pos]))
        #trans_grad_mat = grad_output.transpose(0,1)
        #diag_shared_mat = (torch.diag(ctx.shared_or_not)*1.5).to(ctx.device)
        #shared_grad_output = torch.matmul(trans_grad_mat, diag_shared_mat).transpose(0,1).to(ctx.device)
        #print("modified grad_output : {}".format(shared_grad_output[shared_pos]))
        #output = shared_grad_output.neg() * ctx.alpha

        output = grad_output.neg() * ctx.alpha
        return output, None, None

class GNNConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None,
                 weighted=True):
        super(GNNConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.weighted = weighted

        self.fc_neigh = nn.Linear(self.in_feats, out_feats, bias=bias)
        nn.init.xavier_uniform_(self.fc_neigh.weight,
                                gain=nn.init.calculate_gain('relu'))

    def message_func(self, edges):
        h = edges.src['h']
        if not self.weighted:
            return {'m': h}
        else:
            return {'m': h * edges.data['weight']}

    def forward(self, graph: dgl.DGLHeteroGraph, feat, weight):
        graph = graph.local_var()
        feat_src, feat_dst = self.feat_drop(feat[0]), self.feat_drop(feat[1])
        # node_ids_src, node_ids_dst = node_ids[0], node_ids[1]
        # graph.srcdata['h'], graph.srcdata['id'] = feat_src, node_ids_src
        # graph.dstdata['h'], graph.dstdata['id'] = feat_dst, node_ids_dst
        graph.srcdata['h'], graph.dstdata['h'] = feat_src, feat_dst
        graph.edata['weight'] = weight
        graph.update_all(self.message_func, fn.sum('m', 'neigh'))
        degs = graph.in_degrees().to(feat_dst.device)
        h_neigh = (graph.dstdata['neigh'] +
                   graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
        rst = self.fc_neigh(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
        return rst


class GNN(nn.Module):
    def __init__(
        self,
        in_feats,
        shared_gene,
        human_gene,
        mouse_gene,
        n_hidden,
        n_class,
        n_layer,
        device,
        gene_num,
        activation=None,
        norm=None,
        dropout=0.,
        bias=True,
        weighted=True,
    ):
        super(GNN, self).__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_class = n_class
        self.activation = activation
        self.bias = bias
        self.device = device
        self.layers = nn.ModuleList()
        print("n_layer: {}".format(n_layer))
        self.layers.append(
            GNNConv(in_feats,
                    n_hidden,
                    dropout,
                    norm=norm,
                    bias=self.bias,
                    activation=self.activation,
                    weighted=weighted))
        for _ in range(n_layer - 1):
            self.layers.append(
                GNNConv(n_hidden,
                        n_hidden,
                        dropout,
                        norm=norm,
                        bias=self.bias,
                        activation=self.activation,
                        weighted=weighted))
        self.linear = nn.Linear(n_hidden, n_class)
        nn.init.xavier_uniform_(self.linear.weight,
                                gain=nn.init.calculate_gain('relu'))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(n_hidden, 50))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(50))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(inplace=True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(50, 2))

        self.shared_mat = torch.zeros((gene_num, len(shared_gene))).to(self.device)
        for i in range(len(shared_gene)):
            self.shared_mat[shared_gene[i]][i] = 1.
        self.human_unique_mat = torch.zeros((gene_num, len(human_gene))).to(self.device)
        for i in range(len(human_gene)):
            self.human_unique_mat[human_gene[i]][i] = 1.
        self.mouse_unique_mat = torch.zeros((gene_num, len(mouse_gene))).to(self.device)
        for i in range(len(mouse_gene)):
            self.mouse_unique_mat[mouse_gene[i]][i] = 1.

        embed_dim = 400
        self.shared_embedding = nn.Linear(len(shared_gene), embed_dim)
        self.human_embedding = nn.Linear(len(human_gene), embed_dim)
        self.mouse_embedding = nn.Linear(len(mouse_gene), embed_dim)

    def forward(self, blocks, x, weights, edges, alpha=1):
        h_src = x.float().to(self.device)
        shared_gene_mat = torch.matmul(h_src, self.shared_mat)
        human_unique_gene_mat = torch.matmul(h_src, self.human_unique_mat)
        mouse_unique_gene_mat = torch.matmul(h_src, self.mouse_unique_mat)
        shared_gene_para = self.shared_embedding(shared_gene_mat)
        human_unique_gene_para = self.human_embedding(human_unique_gene_mat)
        mouse_unique_gene_para = self.mouse_embedding(mouse_unique_gene_mat)
        #print("fowarding h_src {}".format(h_src.shape))
        #print('model_forward : {}'.format(x.shape))
        #print('model_shared_or_not_list : {}'.format(len(shared_or_not)))
        h_src = FineGrainedLayerF.apply(shared_gene_para, human_unique_gene_para, mouse_unique_gene_para)
        for i, (layer, block, edge) in enumerate(zip(self.layers, blocks, edges)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst= h_src[:block.number_of_dst_nodes()]
            #print("forwarding h_dst {}".format(h_dst.shape))
#            h_src = torch.nn.Parameter(h_src)
#            self.register_parameter('h_src', h_src)
            #print("list shape : {}".format(shared_or_not.shape))
            #print("dst shape: {}, src shape: {}".format(h_dst.shape, h_src.shape))
            # weight = weights[edge2parent[edge]].to(h_src.device)
            weight = weights[edge].to(h_src.device)
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h_src = layer(block, (h_src, h_dst), weight)

        # domain_x = self.domain_classifier(h_src)
        reverse_x = ReverseLayerF.apply(h_src, alpha) # forward
        domain_x = self.domain_classifier(reverse_x)
        h_x = self.linear(h_src)

        return h_x, domain_x

    def cal_loss(self, pred, gold, smoothing=0.0):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''

        gold = gold.contiguous().view(-1)

        if smoothing != 0.0:
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (
                n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
            # loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')
        return loss
