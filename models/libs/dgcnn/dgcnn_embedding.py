import torch
import torch.nn as nn
from torch.autograd import Variable

from .lib.gnn_lib import gnnlib
from .lib.pytorch_util import weights_init, gnn_spmm


class DGCNN(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=None, k=30, conv1d_channels=None,
                 conv1d_kws=None, conv1d_activation='ReLU'):
        if conv1d_kws is None:
            conv1d_kws = [0, 5]
        if conv1d_channels is None:
            conv1d_channels = [16, 32]
        if latent_dim is None:
            latent_dim = [32, 32, 32, 1]

        print('Initializing DGCNN')
        super(DGCNN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats + num_edge_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))

        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.conv1d_act1 = nn.PReLU(conv1d_channels[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        self.conv1d_act2 = nn.PReLU(conv1d_channels[1])

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)

        act_str = conv1d_activation + ('' if conv1d_activation.find('(') != -1 else '()')
        self.conv1d_activation = eval(f'nn.{act_str}')

        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [torch.Tensor(graph_list[i].degs) + 1 for i in range(len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)

        n2n_sp, e2n_sp, subg_sp = gnnlib.prepare_sparse_matrices(graph_list)

        if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            node_degs = node_degs.cuda()
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
            if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
                edge_feat = edge_feat.cuda()
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        node_degs = Variable(node_degs)

        h = self.sortpooling_embedding(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs)

        return h

    def sortpooling_embedding(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs):
        # If exists edge feature, concatenate to node feature vector
        if edge_feat is not None:
            # input_edge_linear = self.w_e2l(edge_feat)
            input_edge_linear = edge_feat
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        # Graph convolution layers
        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer  # Y = (A + I) * X
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W
            normalized_linear = node_linear.div(node_degs)  # Y = D^-1 * Y
            cur_message_layer = torch.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)

        # Sortpooling layer
        sort_channel = cur_message_layer[:, -1]
        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k, self.total_latent_dim)
        if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
            batch_sortpooling_graphs = batch_sortpooling_graphs.cuda()

        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)
        accum_count = 0
        for i in range(subg_sp.size()[0]):
            to_sort = sort_channel[accum_count: accum_count + graph_sizes[i]]
            k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]
            _, topk_indices = to_sort.topk(k)
            topk_indices += accum_count
            sortpooling_graph = cur_message_layer.index_select(0, topk_indices)
            if k < self.k:
                to_pad = torch.zeros(self.k-k, self.total_latent_dim)
                if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
                    to_pad = to_pad.cuda()

                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += graph_sizes[i]

        # Traditional 1d convlution and dense layers
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_latent_dim))
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = self.conv1d_act1(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = self.conv1d_act2(conv1d_res)

        to_dense = conv1d_res.view(len(graph_sizes), -1)

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = self.conv1d_activation(out_linear)
        else:
            reluact_fp = to_dense

        return self.conv1d_activation(reluact_fp)
