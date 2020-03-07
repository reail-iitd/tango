from src.GNN.oldmodels import *
from src.utils import *


def action2vec(action, num_objects, num_states):
    actionArray = torch.zeros(len(possibleActions))
    actionArray[possibleActions.index(action['name'])] = 1
    predicate1 = torch.zeros(num_objects+1)
    predicate2 = torch.zeros(num_objects+1)
    predicate3 = torch.zeros(num_states+1)
    if len(action['args']) < 1:
        predicate1[-1] = 1
    if len(action['args']) < 2:
        predicate1[object2idx[action['args'][0]]] = 1
        predicate2[-1] = 1
    else:
        if action['args'][1] in object2idx:
            predicate2[object2idx[action['args'][1]]] = 1
            predicate3[-1] = 1
        else:
            predicate2[-1] = 1
            predicate2[possibleStates.index(action['args'][1])] = 1
    return torch.cat((actionArray, predicate1, predicate2, predicate3), 0)

def vec2action(vec, num_objects):
    pass

############################ DGL ############################

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

class GatedHeteroRGCNLayer(nn.Module):
    # Source = https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatedgraphconv.html#GatedGraphConv
    def __init__(self, in_size, out_size, etypes, activation):
        super(GatedHeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({name : nn.Linear(in_size, out_size) for name in etypes})
        self.reduce = nn.Linear(in_size, out_size)
        self.activation = activation
        self.gru = LayerNormGRUCell(out_size, out_size, bias=True)

    def forward(self, G, features):
        funcs = {}; feat = self.activation(self.reduce(features))
        for _ in range(N_TIMESEPS):
            for etype in G.etypes:
                Wh = self.weight[etype](features)
                G.nodes['object'].data['Wh_%s' % etype] = Wh
                funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
            G.multi_update_all(funcs, 'sum')
            feat = self.gru(G.nodes['object'].data['h'], feat)
        return self.activation(feat)

######################################################################################

class DGL_AGCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_classes,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(DGL_AGCN, self).__init__()
        self.name = "GatedHeteroRGCN_Attention_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        # output layer
        self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Sequential()
        self.embed.add_module(
            'fc',
            fc_block(
                PRETRAINED_VECTOR_SIZE,
                n_hidden,
                False,
                nn.Tanh))
        self.fc1 = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_classes)
        self.final = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        goalObjectsVec = self.embed(torch.Tensor(goalObjectsVec))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
        # print(attn_weights)
        scene_embedding = torch.mm(attn_weights.t(), h)
        goal_embed = self.embed(torch.Tensor(goalVec.reshape(1, -1)))
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        h = F.tanh(self.fc1(final_to_decode))
        h = F.tanh(self.fc2(h))
        h = self.final(self.fc3(h))
        return h.flatten()

class DGL_AGCN_Tool(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_classes,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(DGL_AGCN_Tool, self).__init__()
        self.n_classes = n_classes
        self.name = "GatedHeteroRGCN_Attention_Tool_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_classes-1)
        self.p1  = nn.Linear(n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, 1)
        self.final = torch.sigmoid
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_weights = F.softmax(self.activation(self.attention(attn_embedding)), dim=0)
        # print(attn_weights)
        scene_embedding = torch.mm(attn_weights.t(), h)
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        h = self.activation(self.fc1(final_to_decode))
        tools = self.activation(self.fc2(h))
        tools = self.activation(self.fc3(h)).flatten()
        tools = F.softmax(tools, dim=0)
        probNoTool = self.activation(self.p1(h))
        probNoTool = self.activation(self.p2(h))
        probNoTool = torch.sigmoid(self.activation(self.p3(probNoTool))).flatten()
        output = torch.cat(((1-probNoTool)*tools, probNoTool), dim=0)
        return output


class DGL_AGCN_Likelihood(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(DGL_AGCN_Likelihood, self).__init__()
        self.name = "GatedHeteroRGCN_Attention_Likelihood" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        # output layer
        self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Sequential()
        self.embed.add_module(
            'fc',
            fc_block(
                PRETRAINED_VECTOR_SIZE,
                n_hidden,
                False,
                nn.tanh))
        self.fc1 = nn.Linear(n_hidden + n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_hidden)
        self.fc5 = nn.Linear(n_hidden, 1)
        self.final = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        goalObjectsVec = self.embed(torch.Tensor(goalObjectsVec))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
        # print(attn_weights)
        scene_embedding = torch.mm(attn_weights.t(), h)
        goal_embed = self.embed(torch.Tensor(goalVec.reshape(1, -1)))
        scene_and_goal = torch.cat([scene_embedding, goal_embed], 1)
        l = []
        tool_embedding = self.embed(torch.Tensor(tool_vec))
        for i in range(NUMTOOLS):
            final_to_decode = torch.cat([scene_and_goal, tool_embedding[i].view(1, -1)], 1)
            h = torch.tanh(self.fc1(final_to_decode))
            h = torch.tanh(self.fc2(h))
            h = torch.tanh(self.fc3(h))
            h = torch.tanh(self.fc4(h))
            h = self.final(self.fc5(h))
            l.append(h.flatten())
        return torch.stack(l)

class DGL_AGCN_Action(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_states,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(DGL_AGCN_Action, self).__init__()
        self.name = "GatedHeteroRGCN_Attention_Tool_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(possibleActions))
        self.p1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, n_objects)
        self.q1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.q2  = nn.Linear(n_hidden, n_hidden)
        self.q3  = nn.Linear(n_hidden, n_objects)
        self.r1  = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.r2  = nn.Linear(n_hidden, n_hidden)
        self.r3  = nn.Linear(n_hidden, n_states)
        self.activation = nn.LeakyReLU()

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_weights = F.softmax(self.attention(attn_embedding), dim=0)
        # print(attn_weights)
        scene_embedding = torch.mm(attn_weights.t(), h)
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec.reshape(1, -1))))
        final_to_decode = torch.cat([scene_embedding, goal_embed], 1)
        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.activation(self.fc3(action))
        action = F.softmax(action, dim=1)
        pred1 = self.activation(self.p1(final_to_decode))
        pred1 = self.activation(self.p2(pred1))
        pred1 = F.softmax(self.activation(self.p3(pred1)), dim=1)
        pred2 = self.activation(self.q1(final_to_decode))
        pred2 = self.activation(self.q2(pred2))
        pred2 = F.softmax(self.activation(self.q3(pred2)), dim=1)
        pred3 = self.activation(self.r1(final_to_decode))
        pred3 = self.activation(self.r2(pred3))
        pred3 = F.softmax(self.activation(self.r3(pred3)), dim=1)
        return torch.cat((action, pred1, pred2, pred3), 1).flatten()
