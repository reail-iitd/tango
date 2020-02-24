from src.GNN.CONSTANTS import *
from src.GNN.helper import LayerNormGRUCell, fc_block

import torch
import torch.nn as nn

class GraphEncoder(nn.Module):
    def __init__(self):
        super(GraphEncoder, self).__init__()
        # self.object_embeddings = nn.Embedding(NUMOBJECTS, EMBEDDING_DIM)
        self.n_timesteps = N_TIMESEPS
        self.n_edge_types = N_EDGES

        node_init2hidden = nn.Sequential()
        node_init2hidden.add_module(
            'fc',
            fc_block(
                REDUCED_DIMENSION_SIZE + N_STATES + SIZE_AND_POS_SIZE,
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

    def forward(self, adjacency_matrix, node_states, node_ids, node_vectors, node_size_and_pos):

        n_nodes = len(node_ids)
        # node_embeddings = self.object_embeddings(torch.LongTensor(node_ids))
        node_embeddings = torch.Tensor(node_vectors)
        node_embeddings = self.reduce_dimentionality(node_embeddings)

        adjacency_matrix = torch.Tensor(adjacency_matrix)
        node_states = torch.Tensor(node_states)
        node_size_and_pos = torch.Tensor(node_size_and_pos)

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

    def __init__(self):
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
    def __init__(self):
        super(GraphEncoder_Decoder, self).__init__()
        self.encoder = GraphEncoder()
        self.decoder = Decoder()

    def forward(self, adjacency_matrix, node_states, node_ids, node_vectors, node_size_and_pos, goal_vec):
        x = self.encoder.forward(adjacency_matrix, node_states, node_ids, node_vectors, node_size_and_pos)
        goal_vec = torch.Tensor(goal_vec)
        x = x[-1]
        x = torch.cat([x, goal_vec], 0)
        x = x.reshape(1,-1)
        outs = self.decoder.forward(x)
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
