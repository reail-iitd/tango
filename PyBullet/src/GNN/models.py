from src.GNN.CONSTANTS import *
from src.GNN.helper import LayerNormGRUCell, fc_block

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class GraphEncoder(nn.Module):
    def __init__(self, args, goal_bit = False):
        self.args = args
        self.goal_bit = goal_bit
        super(GraphEncoder, self).__init__()
        # self.object_embeddings = nn.Embedding(NUMOBJECTS, EMBEDDING_DIM)
        self.n_timesteps = N_TIMESEPS
        self.n_edge_types = N_EDGES

        node_init2hidden = nn.Sequential()
        node_init2hidden.add_module(
            'fc',
            fc_block(
                REDUCED_DIMENSION_SIZE + N_STATES + SIZE_AND_POS_SIZE + goal_bit,
                GRAPH_HIDDEN,
                False,
                nn.Tanh))

        reduce_dimentionality = nn.Sequential()
        reduce_dimentionality.add_module(
            'fc',
            fc_block(
                PRETRAINED_VECTOR_SIZE,
                REDUCED_DIMENSION_SIZE,
                False,
                nn.Tanh))

        for i in range(N_EDGES):
            hidden2message_in = fc_block(
                GRAPH_HIDDEN, GRAPH_HIDDEN, False, nn.Tanh)
            self.add_module(
                "hidden2message_in_{}".format(i),
                hidden2message_in)

            hidden2message_out = fc_block(
                GRAPH_HIDDEN, GRAPH_HIDDEN, False, nn.Tanh)
            self.add_module(
                "hidden2message_out_{}".format(i),
                hidden2message_out)

        self.node_init2hidden = node_init2hidden
        self.reduce_dimentionality = reduce_dimentionality
        self.propagator = LayerNormGRUCell(2 * self.n_edge_types * GRAPH_HIDDEN, GRAPH_HIDDEN)
        self.hidden2message_in = AttrProxy(self, "hidden2message_in_")
        self.hidden2message_out = AttrProxy(self, "hidden2message_out_")

    def forward(self, adjacency_matrix, node_states, node_ids, node_vectors, node_size_and_pos, goal_bits = None):

        n_nodes = len(node_ids)
        # node_embeddings = self.object_embeddings(torch.LongTensor(node_ids))
        node_embeddings = torch.Tensor(node_vectors)
        node_embeddings = self.reduce_dimentionality(node_embeddings)

        adjacency_matrix = torch.Tensor(adjacency_matrix)
        node_states = torch.Tensor(node_states)
        node_size_and_pos = torch.Tensor(node_size_and_pos)

        if self.goal_bit:
            goal_bits = torch.Tensor(goal_bits)
            node_init = torch.cat((node_embeddings, node_states, node_size_and_pos, goal_bits), 1)
        else:
            node_init = torch.cat((node_embeddings, node_states, node_size_and_pos), 1)
        node_hidden = self.node_init2hidden(node_init)

        adjacency_matrix = adjacency_matrix.float()
        adjacency_matrix_out = adjacency_matrix
        adjacency_matrix_in = adjacency_matrix.permute(0, 2, 1)

        for i in range(self.n_timesteps):
            message_out = []
            for j in range(self.n_edge_types):
                node_state_hidden = self.hidden2message_out[j](node_hidden)
                message_out.append(
                    torch.matmul(
                        adjacency_matrix_out[j],
                        node_state_hidden))

            # concatenate the message from each edges
            message_out = torch.stack(message_out, 1)
            message_out = message_out.view(n_nodes, -1)

            message_in = []
            for j in range(self.n_edge_types):
                node_state_hidden = self.hidden2message_in[j](node_hidden)
                message_in.append(
                    torch.matmul(
                        adjacency_matrix_in[j],
                        node_state_hidden))

            # concatenate the message from each edges
            message_in = torch.stack(message_in, 1)
            message_in = message_in.view(n_nodes, -1)

            message = torch.cat([message_out, message_in], 1)
            node_hidden = self.propagator(message, node_hidden)

        return node_hidden

class Decoder(nn.Module):

    def __init__(self,args):
        self.args = args
        super(Decoder, self).__init__()
        object2logit = nn.Sequential()
        object2logit.add_module(
            'fc1',
            fc_block(
                GRAPH_HIDDEN + NUM_GOALS,
                LOGIT_HIDDEN,
                False,
                nn.Tanh))
        object2logit.add_module(
            'fc2',
            fc_block(
                LOGIT_HIDDEN,
                NUMTOOLS,
                False,
                nn.Sigmoid))
        self.object2logit = object2logit

    def forward(self, x):
        return self.object2logit(x)

class GraphEncoder_Decoder(nn.Module):
    def __init__(self,args):
        self.args = args
        super(GraphEncoder_Decoder, self).__init__()
        self.encoder = GraphEncoder(args)
        self.decoder = Decoder(args)

    def forward(self, adjacency_matrix, node_states, node_ids, node_vectors, node_size_and_pos, goal_vec):
        x = self.encoder.forward(adjacency_matrix, node_states, node_ids, node_vectors, node_size_and_pos)
        goal_vec = torch.Tensor(goal_vec)
        x = x[-1]
        x = torch.cat([x, goal_vec], 0)
        x = x.reshape(1,-1)
        outs = self.decoder.forward(x)
        return outs

class Decoder_New(nn.Module):

    def __init__(self,args):
        self.args = args
        super(Decoder_New, self).__init__()
        object2logit = nn.Sequential()
        object2logit.add_module(
            'fc1',
            fc_block(
                GRAPH_HIDDEN + REDUCED_DIMENSION_SIZE,
                LOGIT_HIDDEN,
                False,
                nn.Tanh))
        object2logit.add_module(
            'fc2',
            fc_block(
                LOGIT_HIDDEN,
                NUMTOOLS,
                False,
                nn.Sigmoid))
        self.object2logit = object2logit

    def forward(self, x):
        return self.object2logit(x)


class GraphAttentionEncoder_Decoder(nn.Module):
    def __init__(self,args):
        self.args = args
        super(GraphAttentionEncoder_Decoder, self).__init__()
        self.encoder = GraphEncoder(args, goal_bit = True)
        self.attn = nn.Linear(GRAPH_HIDDEN + REDUCED_DIMENSION_SIZE, 1)
        # self.attn = nn.Sequential() #Bahdanau attention
        # self.attn.add_module(
        #     'fc1',
        #     fc_block(
        #         GRAPH_HIDDEN + REDUCED_DIMENSION_SIZE,
        #         32,
        #         False,
        #         nn.Tanh))
        # self.attn.add_module(
        #     'fc2',
        #     fc_block(
        #         32,
        #         1,
        #         False,
        #         None))
        self.decoder = Decoder_New(args)

    def forward(self, adjacency_matrix, node_states, node_ids, node_vectors, node_size_and_pos, goal_attention_encoding, goal_encoding, goal_bits):
        goal_encoding = torch.Tensor(goal_encoding)
        goal_encoding = self.encoder.reduce_dimentionality(goal_encoding)
        goal_attention_encoding = torch.Tensor(goal_attention_encoding)
        goal_attention_encoding = self.encoder.reduce_dimentionality(goal_attention_encoding)

        x = self.encoder.forward(adjacency_matrix, node_states, node_ids, node_vectors, node_size_and_pos, goal_bits = goal_bits)
        goal_attention_encoding = goal_attention_encoding.repeat(x.size(0)).view(x.size(0), -1)
        attention_weights = self.attn(torch.cat([x, goal_attention_encoding], 1))
        assert (attention_weights.size(1) == 1)
        attention_weights = F.softmax(attention_weights, dim = 0)
        scene_embedding = torch.mm(attention_weights.t(), x)
        goal_encoding = goal_encoding.view(1,-1)
        final_to_decode = torch.cat([scene_embedding, goal_encoding], 1)
        outs = self.decoder.forward(final_to_decode)
        return outs

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class HeteroRGCNLayer(nn.Module):
    # Source = https://docs.dgl.ai/en/0.4.x/tutorials/hetero/1_basics.html
    def __init__(self, in_size, out_size, etypes, activation):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({name : nn.Linear(in_size, out_size) for name in etypes})
        self.activation = activation

    def forward(self, G, features):
        funcs = {}
        for etype in G.etypes:
            Wh = self.weight[etype](features)
            G.nodes['object'].data['Wh_%s' % etype] = Wh
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        return self.activation(G.nodes['object'].data['h'])

class DGL_GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_classes,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(DGL_GCN, self).__init__()
        self.name = "HeteroRGCN_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(HeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(HeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        # output layer
        self.layers.append(HeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.fc1 = nn.Linear(n_hidden * n_objects, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)
        self.final = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        h = h.flatten()
        h = nn.functional.relu(self.fc1(h))
        h = self.final(self.fc2(h))
        return h

class DGL_AE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 etypes,
                 activation):
        super(DGL_AE, self).__init__()
        self.name = "GCN-AE_" + str(n_hidden) + "_" + str(n_layers)
        self.encoder = nn.ModuleList()
        self.encoder.append(HeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.encoder.append(HeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.decoder = nn.ModuleList()
        for i in range(n_layers - 1):
            self.decoder.append(HeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=nn.functional.tanhshrink))
        self.decoder.append(HeteroRGCNLayer(n_hidden, in_feats, etypes, activation=nn.functional.tanhshrink))

    def encode(self, g, h):
        for i, layer in enumerate(self.encoder):
            h = layer(g, h)
        return h

    def decode(self, g, h):
        for i, layer in enumerate(self.decoder):
            h = layer(g, h)
        return h

    def forward(self, g):
        h = g.ndata['feat']
        h = self.encode(g, h)
        h = self.decode(g, h)
        return h