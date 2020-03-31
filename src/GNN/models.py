from src.GNN.action_models import *

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

class DGL_Simple_Likelihood(nn.Module):
    def __init__(self,
                 in_feats,
                 n_objects,
                 n_hidden,
                 n_classes,
                 n_layers,
                 etypes,
                 activation,
                 dropout,
                 embedding,
                 weighted):
        super(DGL_Simple_Likelihood, self).__init__()
        self.n_classes = n_classes
        self.etypes = etypes
        self.name = "GGCN_Metric_Attn_L_NT_" + ('C_' if 'c' in embedding else '') + ('W_' if weighted else '') + str(n_hidden) + "_" + str(n_layers)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats + n_objects*4, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.attention = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.attention2 = nn.Linear(n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(3 * n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, 1)
        self.p1  = nn.Linear(2 * n_hidden, n_hidden)
        self.p2  = nn.Linear(n_hidden, 1)
        self.final = nn.Sigmoid()
        self.activation = nn.PReLU()

    def forward(self, g, goalVec, goalObjectsVec, tool_vec):
        h = g.ndata['feat']
        edgeMatrices = [g.adjacency_matrix(etype=t) for t in self.etypes]
        edges = torch.cat(edgeMatrices, 1).to_dense()
        h = torch.cat((h, edges), 1)
        for i, layer in enumerate(self.layers):
            h = self.activation(layer(h))
        goalObjectsVec = self.activation(self.embed(goalObjectsVec))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_embedding = self.activation(self.attention(attn_embedding))
        attn_weights = F.softmax(self.activation(self.attention2(attn_embedding)), dim=0)
        # print(attn_weights)
        scene_embedding = torch.mm(attn_weights.t(), h)
        goal_embed = self.activation(self.embed(goalVec))
        scene_and_goal = torch.cat([scene_embedding, goal_embed.view(1,-1)], 1)
        l = []
        tool_embedding = self.activation(self.embed(tool_vec))
        for i in range(NUMTOOLS-1):
            final_to_decode = torch.cat([scene_and_goal, tool_embedding[i].view(1, -1)], 1)
            h = self.activation(self.fc1(final_to_decode))
            h = self.activation(self.fc2(h))
            h = self.activation(self.fc3(h))
            h = self.final(self.fc4(h))
            l.append(h.flatten())
        tools = torch.stack(l)
        probNoTool = self.activation(self.p1(scene_and_goal))
        probNoTool = torch.sigmoid(self.activation(self.p2(probNoTool))).flatten()
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
        self.fc1 = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_classes)
        self.final = nn.Sigmoid()
        self.activation = nn.PReLU()

    def forward(self, g, goalVec, goalObjectsVec, tool_vec):
        h = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
        scene_embedding = torch.sum(h, dim=0).view(1,-1)
        goal_embed = self.activation(self.embed(goalVec))
        h = torch.cat([scene_embedding, goal_embed.view(1,-1)], 1)
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
        self.layers.append(nn.Linear(in_feats + n_objects*4, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_classes)
        self.final = nn.Sigmoid()
        self.activation = nn.PReLU()

    def forward(self, g, goalVec, goalObjectsVec, tool_vec):
        h = g.ndata['feat']
        edgeMatrices = [g.adjacency_matrix(etype=t) for t in self.etypes]
        edges = torch.cat(edgeMatrices, 1).to_dense()
        h = torch.cat((h, edges), 1)
        for i, layer in enumerate(self.layers):
            h = self.activation(layer(h))
        scene_embedding = torch.sum(h, dim=0).view(1,-1)
        goal_embed = self.activation(self.embed(goalVec))
        h = torch.cat([scene_embedding, goal_embed.view(1,-1)], 1)
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
        self.layers.append(nn.Linear(in_feats + n_objects*4, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(n_hidden + n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_classes)
        self.final = nn.Sigmoid()
        self.activation = nn.PReLU()

    def forward(self, g, goalVec, goalObjectsVec, tool_vec):
        h = g.ndata['feat']
        edgeMatrices = [g.adjacency_matrix(etype=t) for t in self.etypes]
        edges = torch.cat(edgeMatrices, 1).to_dense()
        h = torch.cat((h, edges), 1)
        for i, layer in enumerate(self.layers):
            h = self.activation(layer(h))
        goalObjectsVec = self.activation(self.embed(goalObjectsVec))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_weights = F.softmax(self.activation(self.attention(attn_embedding)), dim=0)
        scene_embedding = torch.mm(attn_weights.t(), h)
        goal_embed = self.activation(self.embed(goalVec))
        scene_and_goal = torch.cat([scene_embedding, goal_embed.view(1,-1)], 1)
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
        self.layers.append(nn.Linear(in_feats + n_objects*4, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.attention = nn.Linear(n_hidden + n_hidden, 1)
        self.embed = nn.Linear(PRETRAINED_VECTOR_SIZE, n_hidden)
        self.fc1 = nn.Linear(3 * n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, 1)
        self.final = nn.Sigmoid()
        self.activation = nn.PReLU()

    def forward(self, g, goalVec, goalObjectsVec, tool_vec):
        h = g.ndata['feat']
        edgeMatrices = [g.adjacency_matrix(etype=t) for t in self.etypes]
        edges = torch.cat(edgeMatrices, 1).to_dense()
        h = torch.cat((h, edges), 1)
        for i, layer in enumerate(self.layers):
            h = self.activation(layer(h))
        goalObjectsVec = self.activation(self.embed(goalObjectsVec))
        attn_embedding = torch.cat([h, goalObjectsVec.repeat(h.size(0)).view(h.size(0), -1)], 1)
        attn_weights = F.softmax(self.activation(self.attention(attn_embedding)), dim=0)
        scene_embedding = torch.mm(attn_weights.t(), h)
        goal_embed = self.activation(self.embed(goalVec))
        scene_and_goal = torch.cat([scene_embedding, goal_embed.view(1,-1)], 1)
        l = []
        tool_vec2 = torch.cat((tool_vec, torch.zeros((1, PRETRAINED_VECTOR_SIZE))))
        tool_embedding = self.activation(self.embed(tool_vec2))
        for i in range(NUMTOOLS):
            final_to_decode = torch.cat([scene_and_goal, tool_embedding[i].view(1, -1)], 1)
            h = self.activation(self.fc1(final_to_decode))
            h = self.activation(self.fc2(h))
            h = self.activation(self.fc3(h))
            h = self.final(self.fc4(h))
            l.append(h.flatten())
        tools = torch.stack(l)
        return tools
