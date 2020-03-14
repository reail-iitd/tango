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

class DGL_Simple_Tool(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_classes,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(DGL_Simple_Tool, self).__init__()
        self.n_classes = n_classes
        self.etypes = etypes
        self.name = "Simple_Attention_Tool_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats + 51*4, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_objects * (n_hidden + n_hidden), n_hidden)
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
        edgeMatrices = [g.adjacency_matrix(etype=t) for t in self.etypes]
        edges = torch.cat(edgeMatrices, 1).to_dense()
        h = torch.cat((h, edges), 1)
        for i, layer in enumerate(self.layers):
            h = self.activation(layer(h))
        goalObjectsVec = self.activation(self.embed(torch.Tensor(goalObjectsVec)))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_weights = F.softmax(self.activation(self.attention(attn_embedding)), dim=0)
        # print(attn_weights)
        scene_embedding = torch.mul(attn_weights.expand(h.size(0), h.size(1)), h)
        goal_embed = self.activation(self.embed(torch.Tensor(goalVec)))
        final_to_decode = torch.cat([scene_embedding, goal_embed.repeat(h.size(0)).view(h.size(0), -1)], 1)
        h = self.activation(self.fc1(final_to_decode.view(1, -1)))
        tools = self.activation(self.fc2(h))
        tools = self.activation(self.fc3(h)).flatten()
        tools = torch.sigmoid(tools)
        probNoTool = self.activation(self.p1(h))
        probNoTool = self.activation(self.p2(h))
        probNoTool = torch.sigmoid(self.activation(self.p3(probNoTool))).flatten()
        output = torch.cat(((1-probNoTool)*tools, probNoTool), dim=0)
        return output

class DGL_Simple_Likelihood(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_classes,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(DGL_Simple_Likelihood, self).__init__()
        self.n_classes = n_classes
        self.etypes = etypes
        self.name = "Simple_Attention_Likelihood_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats + 51*4, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_objects * (n_hidden + n_hidden) + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_hidden)
        self.fc5 = nn.Linear(n_hidden, n_hidden)
        self.fc6 = nn.Linear(n_hidden, 1)
        self.p1  = nn.Linear(n_objects * (n_hidden + n_hidden), n_hidden)
        self.p2  = nn.Linear(n_hidden, n_hidden)
        self.p3  = nn.Linear(n_hidden, 1)
        self.final = nn.Sigmoid()
        self.activation = nn.PReLU()

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        edgeMatrices = [g.adjacency_matrix(etype=t) for t in self.etypes]
        edges = torch.cat(edgeMatrices, 1).to_dense()
        h = torch.cat((h, edges), 1)
        for i, layer in enumerate(self.layers):
            h = self.activation(layer(h))
        goalObjectsVec = self.activation(self.embed(goalObjectsVec))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_weights = F.softmax(self.activation(self.attention(attn_embedding)), dim=0)
        # print(attn_weights)
        scene_embedding = torch.mul(attn_weights.expand(h.size(0), h.size(1)), h)
        goal_embed = self.activation(self.embed(goalVec))
        scene_and_goal = torch.cat([scene_embedding, goal_embed.repeat(h.size(0)).view(h.size(0), -1)], 1).view(1, -1)
        l = []
        tool_embedding = self.activation(self.embed(tool_vec))
        for i in range(NUMTOOLS):
            final_to_decode = torch.cat([scene_and_goal, tool_embedding[i].view(1, -1)], 1)
            h = self.activation(self.fc1(final_to_decode))
            h = self.activation(self.fc2(h))
            h = self.activation(self.fc3(h))
            h = self.activation(self.fc4(h))
            h = self.activation(self.fc5(h))
            h = self.final(self.fc6(h))
            l.append(h.flatten())
        tools = torch.stack(l)
        probNoTool = self.activation(self.p1(scene_and_goal))
        probNoTool = self.activation(self.p2(probNoTool))
        probNoTool = torch.sigmoid(self.activation(self.p3(probNoTool))).flatten()
        output = torch.cat(((1-probNoTool)*tools.flatten(), probNoTool), dim=0)
        return output

class GGCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_classes,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN, self).__init__()
        self.name = "GGCN_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(n_hidden, n_hidden, etypes, activation=activation))
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden * n_objects + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_classes)
        self.final = nn.Sigmoid()
        self.activation = nn.PReLU()

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        scene_embedding = h.flatten()
        goal_embed = self.activation(self.embed(goalVec))
        h = torch.cat((scene_embedding, goal_embed))
        h = self.activation(self.fc1(h))
        h = self.activation(self.fc2(h))
        h = self.final(self.fc3(h))
        return h

class GGCN_Metric(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_classes,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_Metric, self).__init__()
        self.etypes = etypes
        self.name = "GGCN_Metric_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats + 51*4, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_objects * n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_classes)
        self.final = nn.Sigmoid()
        self.activation = nn.PReLU()

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        edgeMatrices = [g.adjacency_matrix(etype=t) for t in self.etypes]
        edges = torch.cat(edgeMatrices, 1).to_dense()
        h = torch.cat((h, edges), 1)
        for i, layer in enumerate(self.layers):
            h = self.activation(layer(h))
        scene_embedding = h.flatten()
        goal_embed = self.activation(self.embed(goalVec))
        h = torch.cat((scene_embedding, goal_embed))
        h = self.activation(self.fc1(h))
        h = self.activation(self.fc2(h))
        h = self.final(self.fc3(h))
        return h

class GGCN_Metric_Attn(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_classes,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_Metric_Attn, self).__init__()
        self.etypes = etypes
        self.name = "GGCN_Metric_Attn_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats + 51*4, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_objects * (n_hidden + n_hidden), n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_classes)
        self.final = nn.Sigmoid()
        self.activation = nn.PReLU()

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        edgeMatrices = [g.adjacency_matrix(etype=t) for t in self.etypes]
        edges = torch.cat(edgeMatrices, 1).to_dense()
        h = torch.cat((h, edges), 1)
        for i, layer in enumerate(self.layers):
            h = self.activation(layer(h))
        goalObjectsVec = self.activation(self.embed(goalObjectsVec))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_weights = F.softmax(self.activation(self.attention(attn_embedding)), dim=0)
        scene_embedding = torch.mul(attn_weights.expand(h.size(0), h.size(1)), h)
        goal_embed = self.activation(self.embed(goalVec))
        scene_and_goal = torch.cat([scene_embedding, goal_embed.repeat(h.size(0)).view(h.size(0), -1)], 1).view(1, -1)
        h = self.activation(self.fc1(scene_and_goal))
        h = self.activation(self.fc2(h))
        h = self.final(self.fc3(h))
        return h

class GGCN_Metric_Attn_L(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_classes,
                 n_layers,
                 etypes,
                 activation,
                 dropout):
        super(GGCN_Metric_Attn_L, self).__init__()
        self.etypes = etypes
        self.name = "GGCN_Metric_Attn_L_" + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats + 51*4, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_objects * (n_hidden + n_hidden) + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_hidden)
        self.fc5 = nn.Linear(n_hidden, n_hidden)
        self.fc6 = nn.Linear(n_hidden, 1)
        self.final = nn.Sigmoid()
        self.activation = nn.PReLU()

    def forward(self, g, goalVec, goalObjectsVec):
        h = g.ndata['feat']
        edgeMatrices = [g.adjacency_matrix(etype=t) for t in self.etypes]
        edges = torch.cat(edgeMatrices, 1).to_dense()
        h = torch.cat((h, edges), 1)
        for i, layer in enumerate(self.layers):
            h = self.activation(layer(h))
        goalObjectsVec = self.activation(self.embed(goalObjectsVec))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_weights = F.softmax(self.activation(self.attention(attn_embedding)), dim=0)
        scene_embedding = torch.mul(attn_weights.expand(h.size(0), h.size(1)), h)
        goal_embed = self.activation(self.embed(goalVec))
        scene_and_goal = torch.cat([scene_embedding, goal_embed.repeat(h.size(0)).view(h.size(0), -1)], 1).view(1, -1)
        l = []
        tool_vec2 = torch.cat((tool_vec, torch.ones((1, PRETRAINED_VECTOR_SIZE))))
        tool_embedding = self.activation(self.embed(tool_vec2))
        for i in range(NUMTOOLS):
            final_to_decode = torch.cat([scene_and_goal, tool_embedding[i].view(1, -1)], 1)
            h = self.activation(self.fc1(final_to_decode))
            h = self.activation(self.fc2(h))
            h = self.activation(self.fc3(h))
            h = self.activation(self.fc4(h))
            h = self.activation(self.fc5(h))
            h = self.final(self.fc6(h))
            l.append(h.flatten())
        tools = torch.stack(l)
        return tools